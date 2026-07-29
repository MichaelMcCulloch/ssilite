"""Cross-fitted students trained under discovered input environments.

The estimator is deliberately limited to ``(features, observed_labels)``.  It
uses standardized raw-feature geometry as *exogenous structure* to discover
environments, then trains cluster-focused students and an
environment-balanced student.  Every reported score is out of fold: a
student's prediction for example ``i`` comes from a model whose optimizer
never received ``observed_labels[i]``.

For student ``s`` and example ``i``, let ``a_si`` be the probability assigned
to the observed label, and let ``e(i)`` be its raw-feature cluster.  The main
statistic is

``learnability_i = mean_repeat(a_{e(i),i})``.

Thus a rare example is graded by the student deliberately specialized to its
matched environment.  This is an existential signal without taking a
multiple-comparison-inflated maximum over many students.  The maximum is
retained separately as ``best_environment_support`` for diagnosis.
``shared_corruption_i = 1 - learnability_i`` is high when even the matched
specialist rejects the observed label.  ``base_weights`` normalizes
learnability within each discovered environment and gives every non-empty
environment equal total mass.  It can be passed directly as ``base_weights``
to ``run_experiment``.  Optional bootstrap rounds retrain from paired initial
seeds under the previous round's trust, with damped and maximum-delta-bounded
updates so grade/re-trust/retrain backaction is explicit and inspectable.
Setting ``permanent_distrust`` makes those updates monotone: trust can fall but
never recover in later rounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

type EnsembleMode = Literal["environment", "uniform"]


@dataclass(frozen=True)
class EnvironmentEnsembleConfig:
    """Configuration for the cross-fitted environment ensemble.

    ``uniform`` mode keeps the same number of fits, folds, architecture,
    student-by-student initialization seeds, and optimizer steps as
    ``environment`` mode.  Students have independent initializations, but the
    seed table is paired identically between modes.  Objective weighting is
    therefore the only treatment.  ``permanent_distrust=False`` keeps trust
    updates reversible; enabling it clamps positive updates to zero after
    applying the same damping rule.
    """

    num_environments: int = 3
    num_folds: int = 3
    num_repeats: int = 2
    mode: EnsembleMode = "environment"
    include_balanced_student: bool = True
    focus_mass: float = 0.8
    hidden_dimensions: int = 16
    training_steps: int = 40
    batch_size: int = 64
    learning_rate: float = 0.02
    weight_decay: float = 1e-4
    kmeans_iterations: int = 20
    trust_floor: float = 1e-3
    rounds: int = 1
    trust_damping: float = 1.0
    max_trust_delta: float = 1.0
    permanent_distrust: bool = False
    convergence_tolerance: float = 1e-3
    seed: int = 0
    device: str | torch.device = "input"


@dataclass(frozen=True)
class EnvironmentEnsembleCompute:
    """Exact logical compute counts for one estimator call."""

    model_fits: int
    optimizer_steps: int
    backward_examples: int
    scoring_forward_examples: int
    clustering_distance_evaluations: int


@dataclass(frozen=True)
class EnvironmentEnsembleResult:
    """Per-example cross-fitted statistics and provenance.

    Tensor outputs remain on the selected execution device.  The layout of
    ``oof_probabilities`` and ``label_support_by_student`` is
    ``(repeat, student, example)``.  ``fold_assignments[r, i]`` identifies the
    held-out fold used to score example ``i`` in repeat ``r``.
    ``trust_scores`` is bounded but unnormalized; ``trust_base_weights`` is its
    global normalization; ``equal_environment_base_weights`` isolates pure
    equal-cluster weighting; and ``environment_balanced_trust``/``base_weights``
    combine within-cluster trust with equal total cluster mass.
    """

    config: EnvironmentEnsembleConfig
    environment_ids: Tensor
    fold_assignments: Tensor
    student_initialization_seeds: Tensor
    oof_probabilities: Tensor
    label_support_by_student: Tensor
    matched_environment_support: Tensor
    best_environment_support: Tensor
    learnability: Tensor
    shared_corruption: Tensor
    round_learnability: Tensor
    trust_history: Tensor
    max_trust_delta_history: Tensor
    converged: bool
    trust_scores: Tensor
    trust_base_weights: Tensor
    equal_environment_base_weights: Tensor
    environment_balanced_trust: Tensor
    base_weights: Tensor
    compute: EnvironmentEnsembleCompute


class _Student(nn.Module):
    def __init__(self, input_dimensions: int, hidden_dimensions: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimensions, hidden_dimensions),
            nn.Tanh(),
            nn.Linear(hidden_dimensions, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


def _validate_config(config: EnvironmentEnsembleConfig, num_examples: int) -> None:
    if config.mode not in {"environment", "uniform"}:
        raise ValueError(f"unknown ensemble mode: {config.mode!r}")
    if config.num_environments < 2:
        raise ValueError("num_environments must be at least two")
    if config.num_environments > num_examples:
        raise ValueError("num_environments cannot exceed the number of examples")
    if not 2 <= config.num_folds <= num_examples:
        raise ValueError("num_folds must lie between two and the sample count")
    if config.num_repeats < 1:
        raise ValueError("num_repeats must be positive")
    if not 0.5 < config.focus_mass <= 1:
        raise ValueError("focus_mass must lie in (0.5, 1]")
    if (
        min(
            config.hidden_dimensions,
            config.training_steps,
            config.batch_size,
            config.kmeans_iterations,
            config.rounds,
        )
        < 1
    ):
        raise ValueError("model sizes, steps, and iterations must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if not 0 < config.trust_floor < 1:
        raise ValueError("trust_floor must lie strictly between zero and one")
    if not 0 < config.trust_damping <= 1:
        raise ValueError("trust_damping must lie in (0, 1]")
    if not 0 < config.max_trust_delta <= 1:
        raise ValueError("max_trust_delta must lie in (0, 1]")
    if config.convergence_tolerance < 0:
        raise ValueError("convergence_tolerance must be non-negative")


def _resolve_device(
    features: Tensor,
    requested: str | torch.device,
) -> torch.device:
    if requested == "input":
        device = features.device
    elif requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)


def _standardize(features: Tensor) -> Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    scale = centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    return centered / scale


def _discover_environments(
    features: Tensor,
    *,
    num_environments: int,
    iterations: int,
    generator: torch.Generator,
) -> tuple[Tensor, int]:
    """Run deterministic-seeded farthest-first k-means on raw features."""

    normalized = _standardize(features)
    count = normalized.shape[0]
    first = int(
        torch.randint(count, (), device=features.device, generator=generator).item()
    )
    center_indices = [first]
    nearest_sq = (normalized - normalized[first]).square().sum(dim=1)
    for _ in range(1, num_environments):
        next_index = int(nearest_sq.argmax().item())
        center_indices.append(next_index)
        candidate_sq = (normalized - normalized[next_index]).square().sum(dim=1)
        nearest_sq = torch.minimum(nearest_sq, candidate_sq)

    centers = normalized[
        torch.tensor(center_indices, device=features.device, dtype=torch.long)
    ].clone()
    assignments: Tensor | None = None
    distance_evaluations = count * num_environments
    for _ in range(iterations):
        distances = torch.cdist(normalized, centers).square()
        distance_evaluations += count * num_environments
        next_assignments = distances.argmin(dim=1)
        next_centers = centers.clone()
        for environment in range(num_environments):
            mask = next_assignments == environment
            if torch.any(mask):
                next_centers[environment] = normalized[mask].mean(dim=0)
            else:
                # Re-seed an empty cluster at the currently least represented
                # point, without consulting labels.
                nearest = distances.min(dim=1).values
                next_centers[environment] = normalized[nearest.argmax()]
        converged = assignments is not None and torch.equal(
            assignments,
            next_assignments,
        )
        assignments = next_assignments
        centers = next_centers
        if converged:
            break
    if assignments is None:
        raise RuntimeError("environment discovery performed no iterations")
    return assignments, distance_evaluations


def discover_feature_environments(
    features: Tensor,
    *,
    num_environments: int = 3,
    iterations: int = 20,
    seed: int = 0,
    device: str | torch.device = "input",
) -> Tensor:
    """Discover environments from standardized raw features alone.

    This is the label-free public boundary around the same deterministic
    farthest-first k-means used by :func:`estimate_environment_ensemble`.  It
    performs no student fitting, label scoring, or trust estimation.
    """

    if not isinstance(features, Tensor):
        raise TypeError("features must be a tensor")
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty matrix")
    if not 2 <= num_environments <= features.shape[0]:
        raise ValueError(
            "num_environments must lie between two and the number of examples"
        )
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not torch.all(torch.isfinite(features)):
        raise ValueError("features must be finite")

    execution_device = _resolve_device(features, device)
    working_features = features.detach().to(
        device=execution_device,
        dtype=torch.float32,
    )
    assignments, _ = _discover_environments(
        working_features,
        num_environments=num_environments,
        iterations=iterations,
        generator=_generator(execution_device, seed),
    )
    return assignments


def _make_folds(
    num_examples: int,
    *,
    device: torch.device,
    num_folds: int,
    generator: torch.Generator,
) -> Tensor:
    """Generate label- and environment-independent repeated K-fold splits."""

    folds = torch.empty(num_examples, device=device, dtype=torch.long)
    permutation = torch.randperm(
        num_examples,
        device=device,
        generator=generator,
    )
    folds[permutation] = torch.arange(num_examples, device=device) % num_folds
    return folds


def _initialize_student(model: _Student, generator: torch.Generator) -> None:
    with torch.no_grad():
        for layer in model.modules():
            if not isinstance(layer, nn.Linear):
                continue
            bound = 1 / math.sqrt(layer.in_features)
            layer.weight.uniform_(-bound, bound, generator=generator)
            if layer.bias is not None:
                layer.bias.uniform_(-bound, bound, generator=generator)


def _balanced_probabilities(environment_ids: Tensor) -> Tensor:
    counts = torch.bincount(
        environment_ids,
        minlength=int(environment_ids.max().item()) + 1,
    ).to(dtype=torch.float32)
    probabilities = counts[environment_ids].reciprocal()
    return probabilities / probabilities.sum()


def _focused_probabilities(
    environment_ids: Tensor,
    target_environment: int,
    focus_mass: float,
) -> Tensor:
    focused = environment_ids == target_environment
    focused_count = int(focused.sum().item())
    other_count = environment_ids.numel() - focused_count
    if focused_count == 0 or other_count == 0:
        return _balanced_probabilities(environment_ids)
    probabilities = torch.empty(
        environment_ids.numel(),
        device=environment_ids.device,
        dtype=torch.float32,
    )
    probabilities[focused] = focus_mass / focused_count
    probabilities[~focused] = (1 - focus_mass) / other_count
    return probabilities


def _train_student(
    model: _Student,
    features: Tensor,
    labels: Tensor,
    environment_ids: Tensor,
    trust: Tensor,
    *,
    student_index: int,
    config: EnvironmentEnsembleConfig,
    generator: torch.Generator,
) -> int:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if config.mode == "uniform":
        probabilities = torch.full(
            (labels.numel(),),
            1 / labels.numel(),
            device=labels.device,
            dtype=torch.float32,
        )
    elif student_index < config.num_environments:
        probabilities = _focused_probabilities(
            environment_ids,
            student_index,
            config.focus_mass,
        )
    else:
        probabilities = _balanced_probabilities(environment_ids)

    probabilities = probabilities * trust
    probabilities = probabilities / probabilities.sum()

    backward_examples = 0
    for _ in range(config.training_steps):
        batch_indices = torch.multinomial(
            probabilities,
            config.batch_size,
            replacement=True,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(features[batch_indices])
        loss = F.binary_cross_entropy_with_logits(logits, labels[batch_indices])
        loss.backward()
        optimizer.step()
        backward_examples += batch_indices.numel()
    return backward_examples


def _environment_balanced_base(
    trust_signal: Tensor,
    environment_ids: Tensor,
    *,
    num_environments: int,
    trust_floor: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    trust = trust_signal.clamp(min=trust_floor, max=1)
    trust_base = trust / trust.sum()
    equal_environment_base = torch.zeros_like(trust)
    combined_base = torch.zeros_like(trust)
    nonempty = 0
    for environment in range(num_environments):
        mask = environment_ids == environment
        if not torch.any(mask):
            continue
        nonempty += 1
        equal_environment_base[mask] = 1 / mask.sum()
        combined_base[mask] = trust[mask] / trust[mask].sum()
    equal_environment_base /= nonempty
    combined_base /= nonempty
    return (
        trust,
        trust_base,
        equal_environment_base / equal_environment_base.sum(),
        combined_base / combined_base.sum(),
    )


def _resolve_initial_trust(
    initial_trust: Tensor | None,
    labels: Tensor,
    *,
    trust_floor: float,
) -> Tensor:
    if initial_trust is None:
        return torch.ones_like(labels)
    if not isinstance(initial_trust, Tensor):
        raise TypeError("initial_trust must be a tensor")
    trust = initial_trust.detach().to(device=labels.device, dtype=labels.dtype)
    if trust.shape != labels.shape:
        raise ValueError("initial_trust must contain one value per example")
    if not torch.all(torch.isfinite(trust)):
        raise ValueError("initial_trust must be finite")
    if torch.any((trust < 0) | (trust > 1)):
        raise ValueError("initial_trust must lie in [0, 1]")
    return trust.clamp_min(trust_floor)


def _resolve_environment_ids(
    supplied_ids: Tensor,
    labels: Tensor,
    *,
    num_environments: int,
) -> Tensor:
    if not isinstance(supplied_ids, Tensor):
        raise TypeError("environment_ids must be a tensor")
    ids = supplied_ids.detach().to(device=labels.device)
    if ids.shape != labels.shape:
        raise ValueError("environment_ids must contain one value per example")
    if ids.is_floating_point() and (
        not torch.all(torch.isfinite(ids)) or not torch.all(ids == ids.round())
    ):
        raise ValueError("environment_ids must contain finite integers")
    ids = ids.to(dtype=torch.long)
    if torch.any((ids < 0) | (ids >= num_environments)):
        raise ValueError("environment_ids must lie in the configured range")
    counts = torch.bincount(ids, minlength=num_environments)
    if torch.any(counts == 0):
        raise ValueError("every configured environment must be represented")
    return ids


def estimate_environment_ensemble(
    features: Tensor,
    observed_labels: Tensor,
    *,
    config: EnvironmentEnsembleConfig | None = None,
    initial_trust: Tensor | None = None,
    environment_ids: Tensor | None = None,
) -> EnvironmentEnsembleResult:
    """Estimate cross-fitted learnability from features and observed labels.

    The function never receives clean labels, group membership, or corruption
    indicators.  Feature clustering is transductive, but all label-dependent
    scoring is strictly out of fold.  ``initial_trust`` is an optional
    observed-data prior for iterative bootstrap rounds.  Supplying
    ``environment_ids`` bypasses clustering and supports assignment-permutation
    controls while retaining the same folds and initialization seed table.
    """

    config = config or EnvironmentEnsembleConfig()
    if not isinstance(features, Tensor) or not isinstance(observed_labels, Tensor):
        raise TypeError("features and observed_labels must be tensors")
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty matrix")
    if observed_labels.shape != (features.shape[0],):
        raise ValueError("observed_labels must contain one scalar per example")
    _validate_config(config, features.shape[0])
    if not torch.all(torch.isfinite(features)):
        raise ValueError("features must be finite")
    if not torch.all(torch.isfinite(observed_labels)):
        raise ValueError("observed_labels must be finite")
    if torch.any((observed_labels != 0) & (observed_labels != 1)):
        raise ValueError("observed_labels must be binary")

    device = _resolve_device(features, config.device)
    training_features = features.detach().to(device=device, dtype=torch.float32)
    labels = observed_labels.detach().to(device=device, dtype=torch.float32)
    if environment_ids is None:
        resolved_environment_ids, distance_evaluations = _discover_environments(
            training_features,
            num_environments=config.num_environments,
            iterations=config.kmeans_iterations,
            generator=_generator(device, config.seed + 101),
        )
    else:
        resolved_environment_ids = _resolve_environment_ids(
            environment_ids,
            labels,
            num_environments=config.num_environments,
        )
        distance_evaluations = 0
    environment_ids = resolved_environment_ids
    num_students = config.num_environments + int(config.include_balanced_student)
    fold_assignments = torch.empty(
        (config.num_repeats, labels.numel()),
        device=device,
        dtype=torch.long,
    )
    initialization_seeds = torch.empty(
        (config.num_repeats, config.num_folds, num_students),
        device=device,
        dtype=torch.long,
    )
    for repeat in range(config.num_repeats):
        fold_assignments[repeat] = _make_folds(
            labels.numel(),
            device=device,
            num_folds=config.num_folds,
            generator=_generator(device, config.seed + 1_003 + repeat),
        )
        for fold in range(config.num_folds):
            for student_index in range(num_students):
                initialization_seed = (
                    config.seed
                    + 10_000_019 * repeat
                    + 100_003 * fold
                    + 997 * student_index
                )
                initialization_seeds[repeat, fold, student_index] = initialization_seed

    current_trust = _resolve_initial_trust(
        initial_trust,
        labels,
        trust_floor=config.trust_floor,
    )
    trust_history = torch.empty(
        (config.rounds + 1, labels.numel()),
        device=device,
        dtype=torch.float32,
    )
    trust_history[0] = current_trust
    round_learnability = torch.empty(
        (config.rounds, labels.numel()),
        device=device,
        dtype=torch.float32,
    )
    max_trust_delta_history = torch.empty(
        config.rounds,
        device=device,
        dtype=torch.float32,
    )
    matched_student = environment_ids.view(1, 1, -1).expand(
        config.num_repeats,
        1,
        -1,
    )
    backward_examples = 0
    oof_probabilities: Tensor | None = None
    label_support: Tensor | None = None
    matched_environment_support: Tensor | None = None
    best_environment_support: Tensor | None = None
    learnability: Tensor | None = None

    for round_index in range(config.rounds):
        round_oof_probabilities = torch.empty(
            (config.num_repeats, num_students, labels.numel()),
            device=device,
            dtype=torch.float32,
        )
        for repeat in range(config.num_repeats):
            folds = fold_assignments[repeat]
            for fold in range(config.num_folds):
                train_mask = folds != fold
                train_indices = torch.where(train_mask)[0]
                score_indices = torch.where(~train_mask)[0]
                for student_index in range(num_students):
                    model = _Student(
                        training_features.shape[1],
                        config.hidden_dimensions,
                    ).to(device)
                    initialization_seed = int(
                        initialization_seeds[repeat, fold, student_index].item()
                    )
                    _initialize_student(
                        model,
                        _generator(device, initialization_seed),
                    )
                    backward_examples += _train_student(
                        model,
                        training_features[train_indices],
                        labels[train_indices],
                        environment_ids[train_indices],
                        current_trust[train_indices],
                        student_index=student_index,
                        config=config,
                        generator=_generator(
                            device,
                            config.seed
                            + 1_000_003 * repeat
                            + 10_009 * fold
                            + 97 * student_index,
                        ),
                    )
                    with torch.no_grad():
                        round_oof_probabilities[
                            repeat,
                            student_index,
                            score_indices,
                        ] = model(training_features[score_indices]).sigmoid()

        expanded_labels = labels.view(1, 1, -1)
        round_label_support = torch.where(
            expanded_labels > 0.5,
            round_oof_probabilities,
            1 - round_oof_probabilities,
        )
        round_matched_support = round_label_support.gather(
            dim=1,
            index=matched_student,
        ).squeeze(1)
        round_best_support = round_label_support.max(dim=1).values.mean(dim=0)
        round_score = round_matched_support.mean(dim=0).clamp(0, 1)
        requested_delta = config.trust_damping * (round_score - current_trust)
        if config.permanent_distrust:
            requested_delta = requested_delta.clamp_max(0)
        applied_delta = requested_delta.clamp(
            min=-config.max_trust_delta,
            max=config.max_trust_delta,
        )
        current_trust = (current_trust + applied_delta).clamp(
            min=config.trust_floor,
            max=1,
        )
        round_learnability[round_index] = round_score
        trust_history[round_index + 1] = current_trust
        max_trust_delta_history[round_index] = applied_delta.abs().max()
        oof_probabilities = round_oof_probabilities
        label_support = round_label_support
        matched_environment_support = round_matched_support
        best_environment_support = round_best_support
        learnability = round_score

    if (
        oof_probabilities is None
        or label_support is None
        or matched_environment_support is None
        or best_environment_support is None
        or learnability is None
    ):
        raise RuntimeError("environment ensemble performed no bootstrap rounds")
    shared_corruption = (1 - learnability).clamp(0, 1)
    (
        trust,
        trust_base_weights,
        equal_environment_base_weights,
        base_weights,
    ) = _environment_balanced_base(
        current_trust,
        environment_ids,
        num_environments=config.num_environments,
        trust_floor=config.trust_floor,
    )
    model_fits = config.rounds * config.num_repeats * config.num_folds * num_students
    return EnvironmentEnsembleResult(
        config=config,
        environment_ids=environment_ids,
        fold_assignments=fold_assignments,
        student_initialization_seeds=initialization_seeds,
        oof_probabilities=oof_probabilities,
        label_support_by_student=label_support,
        matched_environment_support=matched_environment_support,
        best_environment_support=best_environment_support,
        learnability=learnability,
        shared_corruption=shared_corruption,
        round_learnability=round_learnability,
        trust_history=trust_history,
        max_trust_delta_history=max_trust_delta_history,
        converged=bool(
            max_trust_delta_history[-1].item() <= config.convergence_tolerance
        ),
        trust_scores=trust,
        trust_base_weights=trust_base_weights,
        equal_environment_base_weights=equal_environment_base_weights,
        environment_balanced_trust=base_weights,
        base_weights=base_weights,
        compute=EnvironmentEnsembleCompute(
            model_fits=model_fits,
            optimizer_steps=model_fits * config.training_steps,
            backward_examples=backward_examples,
            scoring_forward_examples=(
                config.rounds * config.num_repeats * num_students * labels.numel()
            ),
            clustering_distance_evaluations=distance_evaluations,
        ),
    )


class EnvironmentEnsembleEstimator:
    """Callable ``run_experiment`` adapter retaining the last rich result."""

    def __init__(self, config: EnvironmentEnsembleConfig | None = None) -> None:
        self.config = config or EnvironmentEnsembleConfig()
        self.last_result: EnvironmentEnsembleResult | None = None

    def estimate(
        self,
        features: Tensor,
        observed_labels: Tensor,
        *,
        initial_trust: Tensor | None = None,
        environment_ids: Tensor | None = None,
    ) -> EnvironmentEnsembleResult:
        self.last_result = estimate_environment_ensemble(
            features,
            observed_labels,
            config=self.config,
            initial_trust=initial_trust,
            environment_ids=environment_ids,
        )
        return self.last_result

    def __call__(self, features: Tensor, observed_labels: Tensor) -> Tensor:
        """Return normalized base weights for ``run_experiment``."""

        return self.estimate(features, observed_labels).base_weights
