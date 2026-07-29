"""Leakage-safe follow-up to Fable's reflexive trust bootstrap.

The experiment keeps the diagnostic boundary explicit:

* the trust estimator receives only features and observed labels;
* corruption and group metadata are used only after estimation, for reporting;
* trust changes the DRO base measure, while observed raw loss supplies the tilt;
* an oracle clean-label reference remains a ceiling, not an input to the method.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from .acquisition import acquire_for_cluster_coverage
from .bootstrap import (
    BootstrapConfig,
    BootstrapResult,
    average_midrank,
    estimate_cross_fitted_trust,
)
from .data import DatasetSplit, make_support_problem
from .experiment import (
    TrainingDiagnostics,
    _concatenate,
    _resolve_device,
    run_experiment,
)


@dataclass(frozen=True)
class TrustDiagnostics:
    """Post-hoc diagnostics; none of these masks enter trust estimation."""

    clean_majority_mean: float
    clean_minority_mean: float
    corrupted_mean: float
    corruption_auroc: float
    base_minority_mass: float
    base_corruption_mass: float
    rounds_run: int
    converged: bool
    final_max_change: float
    grader_backward_examples: int
    grader_scoring_forward_examples: int


@dataclass(frozen=True)
class FollowupResult:
    """One paired seed of raw, cross-fitted, and oracle controls."""

    seed: int
    device: str
    bootstrap_config: BootstrapConfig
    acquired_count: int
    acquired_minority_count: int
    trust: dict[str, TrustDiagnostics]
    arms: dict[str, TrainingDiagnostics]
    caveat: str


def _binary_auroc(scores: Tensor, positives: Tensor) -> float:
    """Tie-correct AUROC from average ranks."""

    if scores.ndim != 1 or positives.shape != scores.shape:
        raise ValueError("scores and positives must be equal-length vectors")
    positives = positives.to(dtype=torch.bool)
    positive_count = int(positives.sum().item())
    negative_count = scores.numel() - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = average_midrank(scores) * (scores.numel() - 1) + 1
    rank_sum = ranks[positives].to(dtype=torch.float64).sum()
    offset = positive_count * (positive_count + 1) / 2
    return float(((rank_sum - offset) / (positive_count * negative_count)).item())


def _checkpoint_count(config: BootstrapConfig) -> int:
    positions = torch.linspace(0, config.training_steps, config.checkpoints)
    schedule = {int(position.round().item()) for position in positions} | {
        0,
        config.training_steps,
    }
    return len(schedule)


def _trust_diagnostics(
    result: BootstrapResult,
    split: DatasetSplit,
    config: BootstrapConfig,
) -> TrustDiagnostics:
    clean_minority = split.minority & ~split.flipped
    clean_majority = ~split.minority & ~split.flipped
    base = result.trust / result.trust.sum()
    rounds = result.rounds_run
    # Across all folds in one repeat, each example participates in exactly
    # folds - 1 full-batch training objectives and one held-out score.
    backward_examples = (
        rounds
        * config.repeats
        * config.training_steps
        * split.labels.numel()
        * (config.folds - 1)
    )
    scoring_examples = (
        rounds * config.repeats * _checkpoint_count(config) * split.labels.numel()
    )
    return TrustDiagnostics(
        clean_majority_mean=float(result.trust[clean_majority].mean().item()),
        clean_minority_mean=float(result.trust[clean_minority].mean().item()),
        corrupted_mean=float(result.trust[split.flipped].mean().item()),
        corruption_auroc=_binary_auroc(result.suspicion, split.flipped),
        base_minority_mass=float(base[split.minority].sum().item()),
        base_corruption_mass=float(base[split.flipped].sum().item()),
        rounds_run=rounds,
        converged=result.converged,
        final_max_change=result.history[-1].max_trust_change,
        grader_backward_examples=backward_examples,
        grader_scoring_forward_examples=scoring_examples,
    )


def followup_bootstrap_config(seed: int) -> BootstrapConfig:
    """Configuration used for the checked-in follow-up measurements."""

    return BootstrapConfig(
        folds=3,
        repeats=2,
        rounds=5,
        training_steps=80,
        checkpoints=5,
        hidden_dimensions=24,
        seed=1000 + seed,
    )


def run_followup(
    *,
    seed: int = 0,
    steps: int = 120,
    batch_size: int = 32,
    acquisition_count: int = 256,
    learning_rate: float = 0.01,
    bootstrap_config: BootstrapConfig | None = None,
    causal_allocation_arms: bool = False,
    device: torch.device | str = "auto",
) -> FollowupResult:
    """Run the corrected bootstrap against raw-loss and oracle controls."""

    config = bootstrap_config or followup_bootstrap_config(seed)
    execution_device = _resolve_device(device)
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
    supports = {
        "fixed": train,
        "acquired": _concatenate(train, acquired),
    }

    bootstrap_results = {
        name: estimate_cross_fitted_trust(
            split.features,
            split.labels,
            config=config,
        )
        for name, split in supports.items()
    }
    base_weights = {
        name: result.trust / result.trust.sum()
        for name, result in bootstrap_results.items()
    }

    raw = run_experiment(
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        acquisition_count=acquisition_count,
        learning_rate=learning_rate,
        device=execution_device,
        reference_mode="raw_loss",
        causal_allocation_arms=causal_allocation_arms,
    )
    bootstrapped = run_experiment(
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        acquisition_count=acquisition_count,
        learning_rate=learning_rate,
        device=execution_device,
        reference_mode="raw_loss",
        base_weights=base_weights,
        causal_allocation_arms=causal_allocation_arms,
    )
    oracle = run_experiment(
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        acquisition_count=acquisition_count,
        learning_rate=learning_rate,
        device=execution_device,
        reference_mode="oracle",
    )

    arms: dict[str, TrainingDiagnostics] = {}
    for support_name in supports:
        arms[f"erm_{support_name}"] = raw.arms[f"erm_{support_name}"]
        arms[f"raw_{support_name}"] = raw.arms[f"joint_{support_name}"]
        arms[f"bootstrap_{support_name}"] = bootstrapped.arms[f"joint_{support_name}"]
        arms[f"oracle_{support_name}"] = oracle.arms[f"joint_{support_name}"]
        if causal_allocation_arms:
            for prefix in ("joint_q_only", "joint_qp"):
                short = prefix.removeprefix("joint_")
                arms[f"raw_{short}_{support_name}"] = raw.arms[
                    f"{prefix}_{support_name}"
                ]
                arms[f"bootstrap_{short}_{support_name}"] = bootstrapped.arms[
                    f"{prefix}_{support_name}"
                ]

    diagnostics = {
        name: _trust_diagnostics(
            bootstrap_results[name],
            split,
            config,
        )
        for name, split in supports.items()
    }
    return FollowupResult(
        seed=seed,
        device=str(execution_device),
        bootstrap_config=config,
        acquired_count=acquisition_count,
        acquired_minority_count=int(acquired.minority.sum().item()),
        trust=diagnostics,
        arms=arms,
        caveat=(
            "The bootstrap receives only features and observed labels, but its "
            "cross-fitted full-batch graders are compute-heavy. Group and "
            "corruption masks are used only for post-hoc diagnostics. The "
            "oracle arm is a ceiling, not a deployable method."
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the leakage-safe trust-bootstrap follow-up."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
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
    parser.add_argument("--causal-allocation-arms", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(arguments.threads)
    results = [
        run_followup(
            seed=seed,
            steps=arguments.steps,
            batch_size=arguments.batch_size,
            acquisition_count=arguments.acquire,
            learning_rate=arguments.learning_rate,
            causal_allocation_arms=arguments.causal_allocation_arms,
            device=arguments.device,
        )
        for seed in arguments.seeds
    ]
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
