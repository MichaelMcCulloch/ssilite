"""Causal benchmark for expected/unexpected-uncertainty expert births."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace

import torch
from torch import Tensor

from .environment_moe import EnvironmentMoEConfig, train_environment_moe
from .uncertainty_spawning import (
    SpawningMoEConfig,
    SpawningTrainingResult,
    train_spawning_moe,
)

ARM_NAMES = (
    "single",
    "raw_loss",
    "expected_only",
    "unvalidated",
    "joint",
    "environment_oracle",
)


@dataclass(frozen=True)
class UncertaintySplit:
    """Observed tensors plus evaluation-only stratum and clean outcomes."""

    features: Tensor
    labels: Tensor
    clean_labels: Tensor
    stratum: Tensor

    def take(self, indices: Tensor) -> UncertaintySplit:
        return UncertaintySplit(
            features=self.features[indices],
            labels=self.labels[indices],
            clean_labels=self.clean_labels[indices],
            stratum=self.stratum[indices],
        )

    def to(self, device: torch.device | str) -> UncertaintySplit:
        return UncertaintySplit(
            features=self.features.to(device),
            labels=self.labels.to(device),
            clean_labels=self.clean_labels.to(device),
            stratum=self.stratum.to(device),
        )


@dataclass(frozen=True)
class UncertaintySpawningProblem:
    train: UncertaintySplit
    test: UncertaintySplit


@dataclass(frozen=True)
class UncertaintyBenchmarkConfig:
    """Nested support budgets and the three-stratum synthetic problem."""

    budgets: tuple[int, ...] = (256, 512, 1024)
    test_size: int = 3000
    core_dimensions: int = 6
    context_repetitions: int = 3
    rare_fraction: float = 0.12
    stochastic_fraction: float = 0.12
    context_separation: float = 4.0
    context_noise: float = 0.30
    common_target: float = 0.85
    rare_target: float = 0.75
    oracle_training_steps: int = 60
    device: str = "auto"


@dataclass(frozen=True)
class StratumAccuracy:
    overall: float
    common: float
    rare_rule: float
    stochastic_pocket: float


@dataclass(frozen=True)
class CausalArmMetrics:
    accuracy: StratumAccuracy
    active_experts: int
    birth_count: int
    rare_rule_births: int
    stochastic_false_births: int
    surprising_examples: int
    surprise_rate: float
    proposal_count: int
    rare_proposal_count: int
    stochastic_proposal_count: int
    stochastic_rejections: int
    rare_evidence_mean: float | None
    stochastic_evidence_mean: float | None
    calibration_counts: tuple[int, ...]
    expected_uncertainties: tuple[float, ...]
    birth_examples: tuple[int, ...]
    birth_evidence: tuple[float, ...]
    route_counts: tuple[int, ...]
    common_route_counts: tuple[int, ...]
    rare_route_counts: tuple[int, ...]
    stochastic_route_counts: tuple[int, ...]
    compute: dict[str, int]


@dataclass(frozen=True)
class UncertaintyBenchmarkPoint:
    total_labels: int
    common_examples: int
    rare_examples: int
    stochastic_examples: int
    arms: dict[str, CausalArmMetrics]


@dataclass(frozen=True)
class UncertaintyBenchmarkResult:
    seed: int
    device: str
    config: UncertaintyBenchmarkConfig
    spawning_config: SpawningMoEConfig
    points: tuple[UncertaintyBenchmarkPoint, ...]
    labels_to_target: dict[str, int | None]
    caveat: str


def _validate_config(config: UncertaintyBenchmarkConfig) -> None:
    if not config.budgets:
        raise ValueError("budgets must be non-empty")
    if any(budget < 2 for budget in config.budgets):
        raise ValueError("budgets must contain at least two examples")
    if any(
        left >= right
        for left, right in zip(config.budgets, config.budgets[1:], strict=False)
    ):
        raise ValueError("budgets must be strictly increasing")
    if (
        min(
            config.test_size,
            config.core_dimensions,
            config.context_repetitions,
            config.oracle_training_steps,
        )
        < 1
    ):
        raise ValueError("test size, dimensions, and oracle steps must be positive")
    if not 0 < config.rare_fraction < 1:
        raise ValueError("rare_fraction must lie in (0, 1)")
    if not 0 < config.stochastic_fraction < 1:
        raise ValueError("stochastic_fraction must lie in (0, 1)")
    if config.rare_fraction + config.stochastic_fraction >= 1:
        raise ValueError("rare and stochastic fractions must leave common mass")
    if min(config.context_separation, config.context_noise) <= 0:
        raise ValueError("context separation and noise must be positive")
    if any(not 0 <= value <= 1 for value in (config.common_target, config.rare_target)):
        raise ValueError("accuracy targets must lie in [0, 1]")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def _orthogonal_rules(
    dimensions: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    common = torch.randn(dimensions, generator=generator)
    common = common / common.norm()
    rare = torch.randn(dimensions, generator=generator)
    rare = rare - torch.dot(rare, common) * common
    rare = rare / rare.norm()
    return common, rare


def _make_split(
    count: int,
    *,
    common_rule: Tensor,
    rare_rule: Tensor,
    rare_fraction: float,
    stochastic_fraction: float,
    context_separation: float,
    context_noise: float,
    context_repetitions: int,
    generator: torch.Generator,
) -> UncertaintySplit:
    core = torch.randn(count, common_rule.numel(), generator=generator)
    uniforms = torch.rand(count, generator=generator)
    stratum = torch.zeros(count, dtype=torch.long)
    stratum[uniforms < rare_fraction] = 1
    stratum[
        (uniforms >= rare_fraction) & (uniforms < rare_fraction + stochastic_fraction)
    ] = 2
    common_labels = (core @ common_rule >= 0).float()
    rare_labels = (core @ rare_rule >= 0).float()
    random_labels = torch.randint(2, (count,), generator=generator).float()
    clean_labels = torch.where(
        stratum == 1,
        rare_labels,
        torch.where(stratum == 2, random_labels, common_labels),
    )
    labels = clean_labels.clone()

    base_centers = torch.tensor(
        [
            [-context_separation, -context_separation],
            [context_separation, -context_separation],
            [0.0, context_separation],
        ],
        dtype=core.dtype,
    )
    centers = base_centers.repeat_interleave(context_repetitions, dim=1)
    context = centers[stratum] + context_noise * torch.randn(
        count,
        2 * context_repetitions,
        generator=generator,
    )
    return UncertaintySplit(
        features=torch.cat((core, context), dim=1),
        labels=labels,
        clean_labels=clean_labels,
        stratum=stratum,
    )


def make_uncertainty_spawning_problem(
    *,
    train_size: int = 1024,
    test_size: int = 3000,
    core_dimensions: int = 6,
    context_repetitions: int = 3,
    rare_fraction: float = 0.12,
    stochastic_fraction: float = 0.12,
    context_separation: float = 4.0,
    context_noise: float = 0.30,
    seed: int = 0,
) -> UncertaintySpawningProblem:
    """Create common, alternate-rule, and irreducibly random strata."""

    config = UncertaintyBenchmarkConfig(
        budgets=(train_size,),
        test_size=test_size,
        core_dimensions=core_dimensions,
        context_repetitions=context_repetitions,
        rare_fraction=rare_fraction,
        stochastic_fraction=stochastic_fraction,
        context_separation=context_separation,
        context_noise=context_noise,
    )
    _validate_config(config)
    generator = torch.Generator().manual_seed(seed)
    common_rule, rare_rule = _orthogonal_rules(core_dimensions, generator)
    train = _make_split(
        train_size,
        common_rule=common_rule,
        rare_rule=rare_rule,
        rare_fraction=rare_fraction,
        stochastic_fraction=stochastic_fraction,
        context_separation=context_separation,
        context_noise=context_noise,
        context_repetitions=context_repetitions,
        generator=generator,
    )
    test = _make_split(
        test_size,
        common_rule=common_rule,
        rare_rule=rare_rule,
        rare_fraction=rare_fraction,
        stochastic_fraction=stochastic_fraction,
        context_separation=context_separation,
        context_noise=context_noise,
        context_repetitions=context_repetitions,
        generator=generator,
    )
    return UncertaintySpawningProblem(train=train, test=test)


def _accuracy(logits: Tensor, test: UncertaintySplit) -> StratumAccuracy:
    predictions = logits >= 0
    correct = predictions == test.clean_labels.bool()

    def mean_for(stratum: int) -> float:
        mask = test.stratum == stratum
        if not torch.any(mask):
            raise ValueError("test data must contain every evaluation stratum")
        return float(correct[mask].float().mean().item())

    return StratumAccuracy(
        overall=float(correct.float().mean().item()),
        common=mean_for(0),
        rare_rule=mean_for(1),
        stochastic_pocket=mean_for(2),
    )


def _route_counts(
    routes: Tensor,
    strata: Tensor,
    expert_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    def counts(mask: Tensor) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in torch.bincount(
                routes[mask],
                minlength=expert_count,
            ).tolist()
        )

    return (
        counts(torch.ones_like(strata, dtype=torch.bool)),
        counts(strata == 0),
        counts(strata == 1),
        counts(strata == 2),
    )


def _spawning_metrics(
    result: SpawningTrainingResult,
    support: UncertaintySplit,
    test: UncertaintySplit,
) -> CausalArmMetrics:
    rare_births = 0
    stochastic_births = 0
    rare_evidence: list[float] = []
    stochastic_evidence: list[float] = []
    rare_proposals = 0
    stochastic_proposals = 0

    def proposal_stratum(member_example_ids: tuple[int, ...]) -> int:
        member_ids = torch.tensor(
            member_example_ids,
            device=support.stratum.device,
            dtype=torch.long,
        )
        member_strata = support.stratum[member_ids]
        return int(torch.bincount(member_strata, minlength=3).argmax().item())

    for birth in result.births:
        dominant = proposal_stratum(birth.member_example_ids)
        rare_births += dominant == 1
        stochastic_births += dominant == 2
        rare_proposals += dominant == 1
        stochastic_proposals += dominant == 2
        if dominant == 1:
            rare_evidence.append(birth.evidence.context_switch_log_margin)
        elif dominant == 2:
            stochastic_evidence.append(birth.evidence.context_switch_log_margin)
    for rejection in result.rejections:
        if rejection.proposal_id < 0:
            continue
        dominant = proposal_stratum(rejection.member_example_ids)
        rare_proposals += dominant == 1
        stochastic_proposals += dominant == 2
        if rejection.evidence is None:
            continue
        if dominant == 1:
            rare_evidence.append(rejection.evidence.context_switch_log_margin)
        elif dominant == 2:
            stochastic_evidence.append(rejection.evidence.context_switch_log_margin)
    active_count = int(result.predictions.active_expert_mask.sum().item())
    all_counts, common_counts, rare_counts, stochastic_counts = _route_counts(
        result.predictions.routed_expert_indices,
        test.stratum,
        result.config.max_experts,
    )
    return CausalArmMetrics(
        accuracy=_accuracy(result.predictions.routed_logits, test),
        active_experts=active_count,
        birth_count=len(result.births),
        rare_rule_births=rare_births,
        stochastic_false_births=stochastic_births,
        surprising_examples=result.surprising_examples,
        surprise_rate=result.surprising_examples / support.labels.numel(),
        proposal_count=result.proposal_count,
        rare_proposal_count=rare_proposals,
        stochastic_proposal_count=stochastic_proposals,
        stochastic_rejections=stochastic_proposals - stochastic_births,
        rare_evidence_mean=(
            sum(rare_evidence) / len(rare_evidence) if rare_evidence else None
        ),
        stochastic_evidence_mean=(
            sum(stochastic_evidence) / len(stochastic_evidence)
            if stochastic_evidence
            else None
        ),
        calibration_counts=result.calibration_counts,
        expected_uncertainties=result.expected_uncertainties,
        birth_examples=tuple(birth.activation_example for birth in result.births),
        birth_evidence=tuple(
            birth.evidence.unexpected_uncertainty for birth in result.births
        ),
        route_counts=all_counts,
        common_route_counts=common_counts,
        rare_route_counts=rare_counts,
        stochastic_route_counts=stochastic_counts,
        compute=asdict(result.compute),
    )


def _oracle_metrics(
    *,
    support: UncertaintySplit,
    test: UncertaintySplit,
    spawning_config: SpawningMoEConfig,
    training_steps: int,
) -> CausalArmMetrics:
    oracle = train_environment_moe(
        support.features,
        support.labels,
        test.features,
        support.stratum,
        config=EnvironmentMoEConfig(
            hidden_dimensions=spawning_config.hidden_dimensions,
            training_steps=training_steps,
            batch_size=spawning_config.batch_size,
            learning_rate=spawning_config.learning_rate,
            weight_decay=spawning_config.weight_decay,
            seed=spawning_config.seed,
            device=spawning_config.device,
        ),
    )
    routes = oracle.specialist.routed_expert_indices
    all_counts, common_counts, rare_counts, stochastic_counts = _route_counts(
        routes,
        test.stratum,
        3,
    )
    return CausalArmMetrics(
        accuracy=_accuracy(oracle.specialist.routed_logits, test),
        active_experts=3,
        birth_count=0,
        rare_rule_births=0,
        stochastic_false_births=0,
        surprising_examples=0,
        surprise_rate=0,
        proposal_count=0,
        rare_proposal_count=0,
        stochastic_proposal_count=0,
        stochastic_rejections=0,
        rare_evidence_mean=None,
        stochastic_evidence_mean=None,
        calibration_counts=(),
        expected_uncertainties=(),
        birth_examples=(),
        birth_evidence=(),
        route_counts=all_counts,
        common_route_counts=common_counts,
        rare_route_counts=rare_counts,
        stochastic_route_counts=stochastic_counts,
        compute=asdict(oracle.specialist.compute),
    )


def run_uncertainty_spawning_benchmark(
    *,
    seed: int = 0,
    config: UncertaintyBenchmarkConfig | None = None,
    spawning_config: SpawningMoEConfig | None = None,
) -> UncertaintyBenchmarkResult:
    """Run all six paired causal arms over deterministic nested supports."""

    config = config or UncertaintyBenchmarkConfig()
    _validate_config(config)
    execution_device = _resolve_device(config.device)
    max_budget = max(config.budgets)
    problem = make_uncertainty_spawning_problem(
        train_size=max_budget,
        test_size=config.test_size,
        core_dimensions=config.core_dimensions,
        context_repetitions=config.context_repetitions,
        rare_fraction=config.rare_fraction,
        stochastic_fraction=config.stochastic_fraction,
        context_separation=config.context_separation,
        context_noise=config.context_noise,
        seed=seed,
    )
    permutation = torch.randperm(
        max_budget,
        generator=torch.Generator().manual_seed(seed + 15_485_863),
    )
    resolved_spawning_config = replace(
        spawning_config or SpawningMoEConfig(seed=40_000 + seed),
        device=str(execution_device),
    )
    test = problem.test.to(execution_device)
    points: list[UncertaintyBenchmarkPoint] = []
    spawning_modes = (
        "single",
        "raw_loss",
        "expected_only",
        "unvalidated",
        "joint",
    )
    for budget in config.budgets:
        support = problem.train.take(permutation[:budget]).to(execution_device)
        arms: dict[str, CausalArmMetrics] = {}
        for mode in spawning_modes:
            result = train_spawning_moe(
                support.features,
                support.labels,
                test.features,
                mode=mode,  # type: ignore[arg-type]
                config=resolved_spawning_config,
            )
            arms[mode] = _spawning_metrics(result, support, test)
        arms["environment_oracle"] = _oracle_metrics(
            support=support,
            test=test,
            spawning_config=resolved_spawning_config,
            training_steps=config.oracle_training_steps,
        )
        counts = torch.bincount(support.stratum, minlength=3)
        points.append(
            UncertaintyBenchmarkPoint(
                total_labels=budget,
                common_examples=int(counts[0].item()),
                rare_examples=int(counts[1].item()),
                stochastic_examples=int(counts[2].item()),
                arms=arms,
            )
        )

    labels_to_target: dict[str, int | None] = {}
    for arm_name in ARM_NAMES:
        labels_to_target[arm_name] = next(
            (
                point.total_labels
                for point in points
                if point.arms[arm_name].accuracy.common >= config.common_target
                and point.arms[arm_name].accuracy.rare_rule >= config.rare_target
            ),
            None,
        )
    return UncertaintyBenchmarkResult(
        seed=seed,
        device=str(execution_device),
        config=replace(config, device=str(execution_device)),
        spawning_config=resolved_spawning_config,
        points=tuple(points),
        labels_to_target=labels_to_target,
        caveat=(
            "The three-stratum metadata is used only for post-training accuracy, "
            "birth attribution, and the explicit environment-oracle ceiling. "
            "All five spawning arms receive identical observed features, labels, "
            "stream order, initialization seeds, and proposal seeds. Random-pocket "
            "accuracy is descriptive because its labels are irreducible."
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run expected/unexpected-uncertainty MoE spawning controls."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(UncertaintyBenchmarkConfig().budgets),
    )
    parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--proposal-min-support", type=int, default=32)
    parser.add_argument("--challenger-steps", type=int, default=80)
    parser.add_argument("--router-steps", type=int, default=60)
    parser.add_argument("--max-experts", type=int, default=3)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args(argv)
    config = UncertaintyBenchmarkConfig(
        budgets=tuple(arguments.budgets),
        test_size=arguments.test_size,
        device=arguments.device,
    )
    results = []
    for seed in arguments.seeds:
        spawning_config = SpawningMoEConfig(
            max_experts=arguments.max_experts,
            proposal_min_support=arguments.proposal_min_support,
            challenger_steps=arguments.challenger_steps,
            router_steps=arguments.router_steps,
            seed=40_000 + seed,
            device=arguments.device,
        )
        results.append(
            run_uncertainty_spawning_benchmark(
                seed=seed,
                config=config,
                spawning_config=spawning_config,
            )
        )
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
