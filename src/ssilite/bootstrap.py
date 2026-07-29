"""Cross-fitted training-dynamics estimates of per-example trust.

The estimator deliberately has a narrow interface: it sees features and the
labels that a learner would ordinarily observe.  In particular, generator
metadata such as clean labels, corruption flags, or group membership cannot
enter the estimator.

Each example is scored only by models for which it was held out.  Repeated
K-fold splits provide multiple out-of-fold predictions, while fixed splits and
fixed replica initializations make successive trust rounds comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class BootstrapConfig:
    """Controls cross-fitting and the damped trust bootstrap."""

    folds: int = 5
    repeats: int = 2
    rounds: int = 4
    training_steps: int = 160
    checkpoints: int = 5
    hidden_dimensions: int = 24
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    confidence_threshold: float = 0.7
    min_trust: float = 0.05
    ema_decay: float = 0.5
    max_delta: float = 0.2
    convergence_tolerance: float = 1e-3
    loss_weight: float = 0.5
    fit_weight: float = 0.25
    spread_weight: float = 0.25
    seed: int = 0


@dataclass(frozen=True)
class BootstrapRound:
    """Diagnostics for one grade/re-trust round."""

    round_index: int
    trust_before: Tensor
    trust_target: Tensor
    trust: Tensor
    suspicion: Tensor
    heldout_loss: Tensor
    fit_time: Tensor
    replica_spread: Tensor
    oof_probabilities: Tensor
    oof_fit_times: Tensor
    oof_counts: Tensor
    max_trust_change: float


@dataclass(frozen=True)
class BootstrapResult:
    """Final trust and the evidence used to construct it."""

    trust: Tensor
    suspicion: Tensor
    history: tuple[BootstrapRound, ...]
    fold_ids: Tensor
    oof_counts: Tensor
    converged: bool

    @property
    def rounds_run(self) -> int:
        return len(self.history)


def average_midrank(values: Tensor) -> Tensor:
    """Map a finite vector to average ranks in ``[0, 1]``.

    Exact ties receive their shared average rank.  A singleton or a constant
    vector is therefore neutral at ``0.5`` rather than being ordered
    arbitrarily.
    """

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.numel() == 0:
        raise ValueError("values cannot be empty")
    if not torch.all(torch.isfinite(values)):
        raise ValueError("values must be finite")
    rank_dtype = values.dtype if values.is_floating_point() else torch.float32
    if values.numel() == 1:
        return torch.full(
            values.shape,
            0.5,
            device=values.device,
            dtype=rank_dtype,
        )

    sorted_values, order = torch.sort(values, stable=True)
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ends = counts.cumsum(dim=0)
    starts = ends - counts
    midpoints = (starts + ends - 1).to(dtype=rank_dtype) / 2
    midpoints = midpoints / (values.numel() - 1)
    sorted_ranks = torch.repeat_interleave(midpoints, counts)
    ranks = torch.empty(values.shape, device=values.device, dtype=rank_dtype)
    ranks.scatter_(0, order, sorted_ranks)
    return ranks


def make_repeated_folds(
    sample_count: int,
    *,
    folds: int,
    repeats: int,
    seed: int,
) -> Tensor:
    """Create balanced repeated K-fold assignments on the CPU."""

    if sample_count < 2:
        raise ValueError("at least two samples are required")
    if not 2 <= folds <= sample_count:
        raise ValueError("folds must lie between 2 and the sample count")
    if repeats < 2:
        raise ValueError("repeats must be at least 2 for replica spread")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    assignments = torch.empty((repeats, sample_count), dtype=torch.long)
    base = torch.arange(sample_count, dtype=torch.long).remainder(folds)
    for repeat in range(repeats):
        permutation = torch.randperm(sample_count, generator=generator)
        assignments[repeat, permutation] = base
    return assignments


class _DynamicsMLP(nn.Module):
    def __init__(self, input_dimensions: int, hidden_dimensions: int) -> None:
        super().__init__()
        self.network: nn.Module
        if hidden_dimensions == 0:
            self.network = nn.Linear(input_dimensions, 1)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dimensions, hidden_dimensions),
                nn.Tanh(),
                nn.Linear(hidden_dimensions, 1),
            )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


def _validate_config(config: BootstrapConfig, sample_count: int) -> None:
    if not 2 <= config.folds <= sample_count:
        raise ValueError("folds must lie between 2 and the sample count")
    if config.repeats < 2:
        raise ValueError("repeats must be at least 2 for replica spread")
    if config.rounds <= 0:
        raise ValueError("rounds must be positive")
    if config.training_steps <= 0:
        raise ValueError("training_steps must be positive")
    if config.checkpoints < 2:
        raise ValueError("checkpoints must be at least 2")
    if config.hidden_dimensions < 0:
        raise ValueError("hidden_dimensions cannot be negative")
    finite_parameters = (
        config.learning_rate,
        config.weight_decay,
        config.confidence_threshold,
        config.min_trust,
        config.ema_decay,
        config.max_delta,
        config.convergence_tolerance,
        config.loss_weight,
        config.fit_weight,
        config.spread_weight,
    )
    if not all(math.isfinite(parameter) for parameter in finite_parameters):
        raise ValueError("floating-point configuration values must be finite")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if not 0.5 < config.confidence_threshold < 1:
        raise ValueError("confidence_threshold must lie strictly between 0.5 and 1")
    if not 0 < config.min_trust <= 1:
        raise ValueError("min_trust must lie in (0, 1]")
    if not 0 <= config.ema_decay < 1:
        raise ValueError("ema_decay must lie in [0, 1)")
    if not 0 < config.max_delta <= 1:
        raise ValueError("max_delta must lie in (0, 1]")
    if config.convergence_tolerance < 0:
        raise ValueError("convergence_tolerance cannot be negative")
    signal_weights = (
        config.loss_weight,
        config.fit_weight,
        config.spread_weight,
    )
    if any(weight < 0 for weight in signal_weights) or sum(signal_weights) <= 0:
        raise ValueError("signal weights must be non-negative with a positive sum")


def _validate_fold_ids(fold_ids: Tensor, config: BootstrapConfig, count: int) -> None:
    if fold_ids.shape != (config.repeats, count):
        raise ValueError("fold_ids must have shape (repeats, sample_count)")
    if fold_ids.dtype != torch.long:
        raise ValueError("fold_ids must have dtype torch.long")
    if torch.any((fold_ids < 0) | (fold_ids >= config.folds)):
        raise ValueError("fold_ids contain an invalid fold")
    for repeat in range(config.repeats):
        counts = torch.bincount(fold_ids[repeat].cpu(), minlength=config.folds)
        if torch.any(counts == 0):
            raise ValueError("every fold must be non-empty in every repeat")


def _checkpoint_schedule(training_steps: int, checkpoints: int) -> tuple[int, ...]:
    positions = torch.linspace(0, training_steps, checkpoints)
    rounded = {int(position.round().item()) for position in positions}
    return tuple(sorted(rounded | {0, training_steps}))


def _observed_confidence(probability: Tensor, target: Tensor) -> Tensor:
    return torch.where(target.to(dtype=torch.bool), probability, 1 - probability)


def _train_and_score_fold(
    features: Tensor,
    labels: Tensor,
    trust: Tensor,
    train_mask: Tensor,
    heldout_mask: Tensor,
    *,
    config: BootstrapConfig,
    model_seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Train on ``train_mask`` and return held-out probability/loss/fit time."""

    # Initialize on CPU under a private RNG context, then move the model.  This
    # avoids mutating the caller's global RNG and gives replicas identical
    # initializations across bootstrap rounds.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = _DynamicsMLP(features.shape[1], config.hidden_dimensions)
    model = model.to(device=features.device, dtype=features.dtype)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_features = features[train_mask]
    train_labels = labels[train_mask]
    train_trust = trust[train_mask]
    heldout_features = features[heldout_mask]
    heldout_labels = labels[heldout_mask]

    schedule = _checkpoint_schedule(config.training_steps, config.checkpoints)
    first_fit = torch.ones_like(heldout_labels)

    @torch.no_grad()
    def score(completed_steps: int) -> tuple[Tensor, Tensor]:
        probability = model(heldout_features).sigmoid()
        confidence = _observed_confidence(probability, heldout_labels)
        newly_fit = (first_fit == 1) & (confidence >= config.confidence_threshold)
        fit_fraction = completed_steps / config.training_steps
        first_fit[newly_fit] = fit_fraction
        return probability, confidence

    final_probability, _ = score(0)
    checkpoints_set = set(schedule[1:])
    for step in range(1, config.training_steps + 1):
        logits = model(train_features)
        losses = F.binary_cross_entropy_with_logits(
            logits,
            train_labels,
            reduction="none",
        )
        objective = torch.dot(train_trust, losses) / train_trust.sum()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if step in checkpoints_set:
            final_probability, _ = score(step)

    final_loss = F.binary_cross_entropy(
        final_probability,
        heldout_labels,
        reduction="none",
    )
    return final_probability, final_loss, first_fit


def _ranked_suspicion(
    heldout_loss: Tensor,
    fit_time: Tensor,
    replica_spread: Tensor,
    config: BootstrapConfig,
) -> Tensor:
    weights = heldout_loss.new_tensor(
        [config.loss_weight, config.fit_weight, config.spread_weight]
    )
    ranks = torch.stack(
        (
            average_midrank(heldout_loss),
            average_midrank(fit_time),
            average_midrank(replica_spread),
        )
    )
    return (weights[:, None] * ranks).sum(dim=0) / weights.sum()


def _trust_target(suspicion: Tensor, min_trust: float) -> Tensor:
    # Midrank 0.5 means "no comparative evidence".  Only above-neutral
    # suspicion reduces trust; a constant collection of signals leaves trust
    # at one.
    evidence = (2 * suspicion - 1).clamp(0, 1)
    return 1 - (1 - min_trust) * evidence


def estimate_cross_fitted_trust(
    features: Tensor,
    labels: Tensor,
    *,
    config: BootstrapConfig | None = None,
    initial_trust: Tensor | None = None,
    fold_ids: Tensor | None = None,
) -> BootstrapResult:
    """Estimate bounded trust from leakage-free out-of-fold dynamics.

    ``fold_ids`` is optional so experiments can reuse or permute an explicit
    split.  If omitted, it is generated once from ``config.seed`` and reused
    for every trust round.
    """

    config = config or BootstrapConfig()
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("features must have shape (sample_count, dimensions)")
    if features.shape[1] == 0:
        raise ValueError("features must contain at least one dimension")
    count = features.shape[0]
    if labels.shape != (count,):
        raise ValueError("labels must contain one scalar per sample")
    _validate_config(config, count)
    if not torch.all(torch.isfinite(features)):
        raise ValueError("features must be finite")
    if not torch.all(torch.isfinite(labels)):
        raise ValueError("labels must be finite")
    if torch.any((labels != 0) & (labels != 1)):
        raise ValueError("labels must be binary values in {0, 1}")

    if not features.is_floating_point() or features.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        work_features = features.detach().to(dtype=torch.float32)
    else:
        work_features = features.detach()
    work_labels = labels.detach().to(
        device=features.device,
        dtype=work_features.dtype,
    )

    if fold_ids is None:
        folds = make_repeated_folds(
            count,
            folds=config.folds,
            repeats=config.repeats,
            seed=config.seed,
        ).to(features.device)
    else:
        folds = fold_ids.detach().to(device=features.device)
    _validate_fold_ids(folds, config, count)

    if initial_trust is None:
        trust = torch.ones(count, device=features.device, dtype=work_features.dtype)
    else:
        if initial_trust.shape != (count,):
            raise ValueError("initial_trust must contain one value per sample")
        trust = initial_trust.detach().to(
            device=features.device,
            dtype=work_features.dtype,
        )
        if not torch.all(torch.isfinite(trust)):
            raise ValueError("initial_trust must be finite")
        if torch.any((trust < config.min_trust) | (trust > 1)):
            raise ValueError("initial_trust must lie in [min_trust, 1]")
        trust = trust.clone()

    history: list[BootstrapRound] = []
    converged = False
    suspicion = torch.full_like(trust, 0.5)
    oof_counts = torch.zeros(count, device=features.device, dtype=torch.long)

    for round_index in range(config.rounds):
        oof_probabilities = torch.empty(
            (config.repeats, count),
            device=features.device,
            dtype=work_features.dtype,
        )
        oof_losses = torch.empty_like(oof_probabilities)
        oof_fit_times = torch.empty_like(oof_probabilities)
        oof_counts = torch.zeros(
            count,
            device=features.device,
            dtype=torch.long,
        )

        for repeat in range(config.repeats):
            for fold in range(config.folds):
                heldout_mask = folds[repeat] == fold
                train_mask = ~heldout_mask
                probability, loss, fit_time = _train_and_score_fold(
                    work_features,
                    work_labels,
                    trust,
                    train_mask,
                    heldout_mask,
                    config=config,
                    # Common initialization isolates instability due to the
                    # training complement from irrelevant seed-to-seed noise.
                    model_seed=config.seed + 1,
                )
                oof_probabilities[repeat, heldout_mask] = probability
                oof_losses[repeat, heldout_mask] = loss
                oof_fit_times[repeat, heldout_mask] = fit_time
                oof_counts[heldout_mask] += 1

        if torch.any(oof_counts < 2):
            raise RuntimeError("cross-fitting produced fewer than two OOF predictions")

        heldout_loss = oof_losses.mean(dim=0)
        fit_time = oof_fit_times.mean(dim=0)
        replica_spread = oof_probabilities.std(dim=0, unbiased=False)
        suspicion = _ranked_suspicion(
            heldout_loss,
            fit_time,
            replica_spread,
            config,
        )
        target = _trust_target(suspicion, config.min_trust)
        ema_proposal = config.ema_decay * trust + (1 - config.ema_decay) * target
        delta = (ema_proposal - trust).clamp(
            min=-config.max_delta,
            max=config.max_delta,
        )
        next_trust = (trust + delta).clamp(config.min_trust, 1)
        max_change = float(delta.abs().max().item())

        history.append(
            BootstrapRound(
                round_index=round_index,
                trust_before=trust.detach().clone(),
                trust_target=target.detach().clone(),
                trust=next_trust.detach().clone(),
                suspicion=suspicion.detach().clone(),
                heldout_loss=heldout_loss.detach().clone(),
                fit_time=fit_time.detach().clone(),
                replica_spread=replica_spread.detach().clone(),
                oof_probabilities=oof_probabilities.detach().clone(),
                oof_fit_times=oof_fit_times.detach().clone(),
                oof_counts=oof_counts.detach().clone(),
                max_trust_change=max_change,
            )
        )
        trust = next_trust.detach()
        if max_change <= config.convergence_tolerance:
            converged = True
            break

    return BootstrapResult(
        trust=trust.detach().clone(),
        suspicion=suspicion.detach().clone(),
        history=tuple(history),
        fold_ids=folds.detach().clone(),
        oof_counts=oof_counts.detach().clone(),
        converged=converged,
    )
