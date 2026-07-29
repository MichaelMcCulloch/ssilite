"""Paired mechanism test for environment-decorrelated students.

This experiment deliberately excludes adaptive sampling and mixed precision.
It asks whether changing the *objectives* of out-of-fold students supplies a
learnability signal that an ordinary seed ensemble does not, and whether a
bounded adversary over the resulting label-free environments improves a fresh
learner.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .acquisition import acquire_for_cluster_coverage
from .adversarial import (
    EnvironmentAdversary,
    EnvironmentAdversaryConfig,
)
from .bootstrap import average_midrank
from .data import DatasetSplit, SupportProblem, make_support_problem
from .environment_ensemble import (
    EnvironmentEnsembleConfig,
    EnvironmentEnsembleResult,
    estimate_environment_ensemble,
)
from .environment_mixture import (
    EnvironmentMixtureConfig,
    EnvironmentMixtureResult,
    train_environment_mixture,
)
from .experiment import _concatenate, _resolve_device
from .model import Accuracy, MechanismMLP, accuracy_by_group

type CorruptionMode = Literal["independent", "minority_systematic"]


@dataclass(frozen=True)
class EnsembleEvidence:
    """Post-hoc cohort evidence; metadata never enters the estimator."""

    clean_majority_support: float
    clean_minority_support: float
    corrupted_support: float
    clean_minority_minus_corrupted: float
    rare_vs_corrupt_auroc: float
    clean_minority_retention_at_07: float
    corrupted_acceptance_at_07: float
    mean_rescue_gap_clean_minority: float
    mean_rescue_gap_corrupted: float
    converged: bool
    final_max_trust_delta: float
    cluster_sizes: tuple[int, ...]
    cluster_minority_fractions: tuple[float, ...]
    cluster_corruption_fractions: tuple[float, ...]
    model_fits: int
    backward_examples: int


@dataclass(frozen=True)
class LearnerEvidence:
    """A fresh learner's clean-test result and final training objective mass."""

    accuracy: Accuracy
    balanced_log_loss: float
    final_minority_mass: float
    final_corruption_mass: float
    final_environment_weights: tuple[float, ...] | None
    backward_examples: int


@dataclass(frozen=True)
class MixtureEvidence:
    """Clean-test evidence for equal-compute final student populations."""

    accuracy: Accuracy
    balanced_log_loss: float
    mean_student_prediction_correlation: float
    model_fits: int
    backward_examples: int


@dataclass(frozen=True)
class ReconstructionResult:
    seed: int
    device: str
    label_noise: float
    corruption_mode: CorruptionMode
    acquired_count: int
    acquired_minority_count: int
    acquired_corrupted_count: int
    ensemble_config: EnvironmentEnsembleConfig
    ensembles: dict[str, EnsembleEvidence]
    learners: dict[str, LearnerEvidence]
    mixtures: dict[str, MixtureEvidence]
    caveat: str


def _cohort_mean(values: Tensor, mask: Tensor) -> float:
    if not torch.any(mask):
        return float("nan")
    return float(values[mask].mean().item())


def _rare_vs_corrupt_auroc(
    scores: Tensor,
    clean_minority: Tensor,
    corrupted: Tensor,
) -> float:
    subset = clean_minority | corrupted
    positive_count = int(clean_minority.sum().item())
    negative_count = int(corrupted.sum().item())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    subset_scores = scores[subset]
    positives = clean_minority[subset]
    ranks = average_midrank(subset_scores) * (subset_scores.numel() - 1) + 1
    rank_sum = ranks[positives].to(dtype=torch.float64).sum()
    offset = positive_count * (positive_count + 1) / 2
    return float(((rank_sum - offset) / (positive_count * negative_count)).item())


def _fraction_at_threshold(values: Tensor, mask: Tensor, threshold: float) -> float:
    if not torch.any(mask):
        return float("nan")
    return float((values[mask] >= threshold).to(dtype=torch.float32).mean().item())


def _ensemble_evidence(
    result: EnvironmentEnsembleResult,
    split: DatasetSplit,
) -> EnsembleEvidence:
    clean_minority = split.minority & ~split.flipped
    clean_majority = ~split.minority & ~split.flipped
    support = result.learnability
    student_mean = result.label_support_by_student.mean(dim=(0, 1))
    rescue_gap = result.best_environment_support - student_mean
    minority_support = _cohort_mean(support, clean_minority)
    corrupted_support = _cohort_mean(support, split.flipped)
    environment_count = result.config.num_environments
    sizes: list[int] = []
    minority_fractions: list[float] = []
    corruption_fractions: list[float] = []
    for environment in range(environment_count):
        mask = result.environment_ids == environment
        sizes.append(int(mask.sum().item()))
        minority_fractions.append(_cohort_mean(split.minority.float(), mask))
        corruption_fractions.append(_cohort_mean(split.flipped.float(), mask))

    return EnsembleEvidence(
        clean_majority_support=_cohort_mean(support, clean_majority),
        clean_minority_support=minority_support,
        corrupted_support=corrupted_support,
        clean_minority_minus_corrupted=minority_support - corrupted_support,
        rare_vs_corrupt_auroc=_rare_vs_corrupt_auroc(
            support,
            clean_minority,
            split.flipped,
        ),
        clean_minority_retention_at_07=_fraction_at_threshold(
            support,
            clean_minority,
            0.7,
        ),
        corrupted_acceptance_at_07=_fraction_at_threshold(
            support,
            split.flipped,
            0.7,
        ),
        mean_rescue_gap_clean_minority=_cohort_mean(rescue_gap, clean_minority),
        mean_rescue_gap_corrupted=_cohort_mean(rescue_gap, split.flipped),
        converged=result.converged,
        final_max_trust_delta=float(result.max_trust_delta_history[-1].item()),
        cluster_sizes=tuple(sizes),
        cluster_minority_fractions=tuple(minority_fractions),
        cluster_corruption_fractions=tuple(corruption_fractions),
        model_fits=result.compute.model_fits,
        backward_examples=result.compute.backward_examples,
    )


def _balanced_log_loss(model: nn.Module, split: DatasetSplit) -> float:
    with torch.no_grad():
        losses = F.binary_cross_entropy_with_logits(
            model(split.features),
            split.clean_labels,
            reduction="none",
        )
        majority = losses[~split.minority].mean()
        minority = losses[split.minority].mean()
    return float(((majority + minority) / 2).item())


def _prediction_evidence(
    logits: Tensor,
    split: DatasetSplit,
    *,
    correlation: Tensor,
    model_fits: int,
    backward_examples: int,
) -> MixtureEvidence:
    with torch.no_grad():
        correct = (logits >= 0) == split.clean_labels.to(dtype=torch.bool)
        losses = F.binary_cross_entropy_with_logits(
            logits,
            split.clean_labels,
            reduction="none",
        )

        def group_accuracy(mask: Tensor) -> float:
            return float(correct[mask].to(dtype=torch.float32).mean().item())

        accuracy = Accuracy(
            overall=float(correct.to(dtype=torch.float32).mean().item()),
            majority=group_accuracy(~split.minority),
            minority=group_accuracy(split.minority),
        )
        balanced_loss = (
            losses[~split.minority].mean() + losses[split.minority].mean()
        ) / 2
    return MixtureEvidence(
        accuracy=accuracy,
        balanced_log_loss=float(balanced_loss.item()),
        mean_student_prediction_correlation=float(correlation.item()),
        model_fits=model_fits,
        backward_examples=backward_examples,
    )


def _mixture_evidence(
    result: EnvironmentMixtureResult,
    test: DatasetSplit,
) -> dict[str, MixtureEvidence]:
    ordinary = result.ordinary
    specialist = result.specialist
    test_indices = torch.arange(
        test.labels.numel(),
        device=ordinary.logits.device,
    )
    ordinary_routed_logits = ordinary.logits[
        specialist.routed_student_indices,
        test_indices,
    ]
    return {
        "ordinary_mean": _prediction_evidence(
            ordinary.mean_logits,
            test,
            correlation=ordinary.diagnostics.mean_off_diagonal_correlation,
            model_fits=ordinary.compute.model_fits,
            backward_examples=ordinary.compute.backward_examples,
        ),
        "ordinary_routed": _prediction_evidence(
            ordinary_routed_logits,
            test,
            correlation=ordinary.diagnostics.mean_off_diagonal_correlation,
            model_fits=ordinary.compute.model_fits,
            backward_examples=ordinary.compute.backward_examples,
        ),
        "specialist_mean": _prediction_evidence(
            specialist.mean_logits,
            test,
            correlation=specialist.diagnostics.mean_off_diagonal_correlation,
            model_fits=specialist.compute.model_fits,
            backward_examples=specialist.compute.backward_examples,
        ),
        "specialist_routed": _prediction_evidence(
            specialist.routed_logits,
            test,
            correlation=specialist.diagnostics.mean_off_diagonal_correlation,
            model_fits=specialist.compute.model_fits,
            backward_examples=specialist.compute.backward_examples,
        ),
    }


def _train_full_support(
    model: nn.Module,
    split: DatasetSplit,
    *,
    steps: int,
    learning_rate: float,
    fixed_weights: Tensor | None = None,
    adversary: EnvironmentAdversary | None = None,
) -> tuple[Tensor, Tensor | None]:
    if (fixed_weights is None) == (adversary is None):
        raise ValueError("supply exactly one fixed objective or adversary")
    if fixed_weights is not None:
        weights = fixed_weights.detach().to(
            device=split.labels.device,
            dtype=split.labels.dtype,
        )
        weights = weights / weights.sum()
    else:
        weights = torch.full_like(split.labels, 1 / split.labels.numel())

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    final_environment_weights: Tensor | None = None
    for _ in range(steps):
        losses = F.binary_cross_entropy_with_logits(
            model(split.features),
            split.labels,
            reduction="none",
        )
        if adversary is not None:
            allocation = adversary.update(losses.detach())
            weights = allocation.example_weights
            final_environment_weights = allocation.environment_weights
        optimizer.zero_grad(set_to_none=True)
        torch.dot(weights.detach(), losses).backward()
        optimizer.step()
    return weights.detach(), final_environment_weights


def _learner_evidence(
    model: nn.Module,
    train: DatasetSplit,
    test: DatasetSplit,
    weights: Tensor,
    environment_weights: Tensor | None,
    *,
    steps: int,
) -> LearnerEvidence:
    return LearnerEvidence(
        accuracy=accuracy_by_group(model, test),
        balanced_log_loss=_balanced_log_loss(model, test),
        final_minority_mass=float(weights[train.minority].sum().item()),
        final_corruption_mass=float(weights[train.flipped].sum().item()),
        final_environment_weights=(
            None
            if environment_weights is None
            else tuple(float(value) for value in environment_weights.cpu())
        ),
        backward_examples=steps * train.labels.numel(),
    )


def _permuted_environments(
    environment_ids: Tensor,
    *,
    seed: int,
) -> Tensor:
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


def _systematically_flip_minority(split: DatasetSplit) -> DatasetSplit:
    return DatasetSplit(
        features=split.features,
        labels=torch.where(split.minority, 1 - split.labels, split.labels),
        clean_labels=split.clean_labels,
        minority=split.minority,
        flipped=split.minority.clone(),
    )


def _apply_corruption_mode(
    problem: SupportProblem,
    mode: CorruptionMode,
) -> SupportProblem:
    if mode == "independent":
        return problem
    if mode != "minority_systematic":
        raise ValueError(f"unknown corruption mode: {mode!r}")
    return SupportProblem(
        train=_systematically_flip_minority(problem.train),
        reservoir=_systematically_flip_minority(problem.reservoir),
        test=problem.test,
    )


def reconstruction_ensemble_config(
    seed: int,
    *,
    device: torch.device | str,
) -> EnvironmentEnsembleConfig:
    return EnvironmentEnsembleConfig(
        num_environments=3,
        num_folds=3,
        num_repeats=2,
        hidden_dimensions=16,
        training_steps=40,
        batch_size=64,
        rounds=3,
        trust_damping=0.5,
        max_trust_delta=0.2,
        convergence_tolerance=0.01,
        seed=20_000 + seed,
        device=device,
    )


def run_reconstruction(
    *,
    seed: int = 0,
    acquisition_count: int = 256,
    label_noise: float = 0.04,
    corruption_mode: CorruptionMode = "independent",
    learner_steps: int = 120,
    learning_rate: float = 0.01,
    ensemble_config: EnvironmentEnsembleConfig | None = None,
    device: torch.device | str = "auto",
) -> ReconstructionResult:
    """Run ordinary, environment-specialist, and environment-adversary arms."""

    if learner_steps < 1:
        raise ValueError("learner_steps must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    execution_device = _resolve_device(device)
    generator_noise = label_noise if corruption_mode == "independent" else 0.0
    problem = make_support_problem(
        seed=seed,
        label_noise=generator_noise,
        test_minority_fraction=0.5,
    )
    problem = _apply_corruption_mode(problem, corruption_mode)
    acquisition = acquire_for_cluster_coverage(
        problem.train.features,
        problem.reservoir.features,
        acquisition_count,
        num_clusters=4,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    acquired = problem.reservoir.take(acquisition.indices).to(execution_device)
    train = _concatenate(problem.train, problem.reservoir.take(acquisition.indices))
    train = train.to(execution_device)
    test = problem.test.to(execution_device)

    environment_config = ensemble_config or reconstruction_ensemble_config(
        seed,
        device=execution_device,
    )
    environment_config = replace(
        environment_config,
        mode="environment",
        device=str(execution_device),
    )
    environment = estimate_environment_ensemble(
        train.features,
        train.labels,
        config=environment_config,
    )
    uniform = estimate_environment_ensemble(
        train.features,
        train.labels,
        config=replace(environment_config, mode="uniform"),
        environment_ids=environment.environment_ids,
    )
    permuted = estimate_environment_ensemble(
        train.features,
        train.labels,
        config=environment_config,
        environment_ids=_permuted_environments(
            environment.environment_ids,
            seed=90_000 + seed,
        ),
    )
    mixture_config = EnvironmentMixtureConfig(
        training_steps=80,
        seed=40_000 + seed,
        device=execution_device,
    )
    final_mixture = train_environment_mixture(
        train.features,
        train.labels,
        test.features,
        environment.environment_ids,
        config=mixture_config,
        # Give both equal-compute final arms the same OOF-derived noise
        # filter. Their only treatment is then uniform versus environment-
        # specialized training, plus routing for the final specialist arm.
        train_trust=environment.trust_scores,
    )
    permuted_mixture = train_environment_mixture(
        train.features,
        train.labels,
        test.features,
        permuted.environment_ids,
        config=mixture_config,
        # Keep trust fixed so only the environment assignment is destroyed.
        train_trust=environment.trust_scores,
    )

    torch.manual_seed(seed)
    template = MechanismMLP(train.features.shape[1]).to(execution_device)
    initial_state = copy.deepcopy(template.state_dict())
    adversary_config = EnvironmentAdversaryConfig()
    arm_specs: dict[
        str,
        tuple[Tensor | None, EnvironmentAdversary | None],
    ] = {
        "erm": (torch.full_like(train.labels, 1 / train.labels.numel()), None),
        "ordinary_ensemble_trust": (uniform.trust_base_weights, None),
        "environment_ensemble_trust": (environment.trust_base_weights, None),
        "environment_balanced": (environment.base_weights, None),
        "environment_adversary": (
            None,
            EnvironmentAdversary(
                environment.environment_ids,
                environment.trust_scores,
                adversary_config,
            ),
        ),
        "oracle_environment_adversary": (
            None,
            EnvironmentAdversary(
                train.minority.to(dtype=torch.long),
                torch.where(
                    train.flipped,
                    train.labels.new_tensor(1e-3),
                    train.labels.new_tensor(1.0),
                ),
                adversary_config,
            ),
        ),
    }
    learners: dict[str, LearnerEvidence] = {}
    for name, (fixed_weights, adversary) in arm_specs.items():
        model = MechanismMLP(train.features.shape[1]).to(execution_device)
        model.load_state_dict(initial_state)
        final_weights, final_environment_weights = _train_full_support(
            model,
            train,
            steps=learner_steps,
            learning_rate=learning_rate,
            fixed_weights=fixed_weights,
            adversary=adversary,
        )
        learners[name] = _learner_evidence(
            model,
            train,
            test,
            final_weights,
            final_environment_weights,
            steps=learner_steps,
        )

    mixtures = _mixture_evidence(final_mixture, test)
    mixtures.update(
        {
            f"permuted_{name}": evidence
            for name, evidence in _mixture_evidence(permuted_mixture, test).items()
        }
    )
    return ReconstructionResult(
        seed=seed,
        device=str(execution_device),
        label_noise=generator_noise,
        corruption_mode=corruption_mode,
        acquired_count=acquisition_count,
        acquired_minority_count=int(acquired.minority.sum().item()),
        acquired_corrupted_count=int(acquired.flipped.sum().item()),
        ensemble_config=environment_config,
        ensembles={
            "ordinary": _ensemble_evidence(uniform, train),
            "environment": _ensemble_evidence(environment, train),
            "permuted_environment": _ensemble_evidence(permuted, train),
        },
        learners=learners,
        mixtures=mixtures,
        caveat=(
            "Environment discovery is transductive and uses separable raw-feature "
            "geometry. Labels are scored out of fold, but feature-dependent "
            "systematic corruption remains unidentifiable from a genuine rare "
            "mechanism. Internal students are not independent statistical units."
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Test ordinary versus environment-decorrelated student ensembles."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--acquire", type=int, default=256)
    parser.add_argument("--label-noise", type=float, default=0.04)
    parser.add_argument(
        "--corruption-mode",
        choices=("independent", "minority_systematic"),
        default="independent",
    )
    parser.add_argument("--learner-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args(argv)
    results = [
        run_reconstruction(
            seed=seed,
            acquisition_count=arguments.acquire,
            label_noise=arguments.label_noise,
            corruption_mode=arguments.corruption_mode,
            learner_steps=arguments.learner_steps,
            learning_rate=arguments.learning_rate,
            device=arguments.device,
        )
        for seed in arguments.seeds
    ]
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
