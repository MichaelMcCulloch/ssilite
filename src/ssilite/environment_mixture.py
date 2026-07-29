"""Equal-compute final ensembles over discovered train environments.

This module is downstream of environment discovery.  It deliberately accepts
only train features, observed labels, test features, and discovered train
environment identifiers (plus optional observed-data trust).  In particular,
clean labels and hidden group or corruption metadata cannot enter training.

Two final-model arms are trained from the same independent initialization
seeds and with exactly the same minibatch and optimizer-step budgets:

* ``ordinary`` trains every student with the ordinary empirical distribution.
* ``specialist`` assigns one student to each environment and, optionally, one
  student to an environment-balanced objective.

The specialist arm exposes both an ordinary ensemble mean and a routed
prediction.  Routing standardizes test features with train statistics and
selects the nearest train-environment center.  Thus no test labels or hidden
metadata are needed to choose a specialist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class EnvironmentMixtureConfig:
    """Training and architecture settings for the paired final ensembles."""

    include_balanced_student: bool = True
    focus_mass: float = 0.9
    hidden_dimensions: int = 16
    training_steps: int = 80
    batch_size: int = 64
    learning_rate: float = 0.02
    weight_decay: float = 1e-4
    seed: int = 0
    device: str | torch.device = "input"


@dataclass(frozen=True)
class EnvironmentMixtureCompute:
    """Exact logical compute used by one arm."""

    model_fits: int
    optimizer_steps: int
    backward_examples: int
    train_diagnostic_forward_examples: int
    test_forward_examples: int

    @property
    def total_forward_examples(self) -> int:
        """Examples sent through models, including backward and evaluation."""

        return (
            self.backward_examples
            + self.train_diagnostic_forward_examples
            + self.test_forward_examples
        )


@dataclass(frozen=True)
class StudentDiagnostics:
    """Observed-train error and pairwise prediction correlation by student."""

    error_rates: Tensor
    prediction_correlation: Tensor
    mean_off_diagonal_correlation: Tensor


@dataclass(frozen=True)
class EnsemblePredictions:
    """Individual and probability-averaged predictions for an ensemble.

    ``logits`` and ``probabilities`` have shape ``(student, test_example)``.
    ``mean_logits`` is the logit transform of ``mean_probabilities`` so the
    two ensemble-level representations denote exactly the same prediction.
    """

    logits: Tensor
    probabilities: Tensor
    mean_logits: Tensor
    mean_probabilities: Tensor
    model_seeds: Tensor
    diagnostics: StudentDiagnostics
    compute: EnvironmentMixtureCompute


@dataclass(frozen=True)
class SpecialistPredictions(EnsemblePredictions):
    """Specialist mean plus nearest-environment routed predictions."""

    student_environment_ids: Tensor
    routed_student_indices: Tensor
    routed_logits: Tensor
    routed_probabilities: Tensor


@dataclass(frozen=True)
class EnvironmentMixtureResult:
    """Paired ordinary and specialist final-ensemble outputs."""

    config: EnvironmentMixtureConfig
    ordinary: EnsemblePredictions
    specialist: SpecialistPredictions
    routing_environment_ids: Tensor
    environment_centers: Tensor
    feature_mean: Tensor
    feature_scale: Tensor


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


def _validate_config(config: EnvironmentMixtureConfig) -> None:
    if config.hidden_dimensions < 1:
        raise ValueError("hidden_dimensions must be positive")
    if config.training_steps < 1:
        raise ValueError("training_steps must be positive")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0.5 < config.focus_mass <= 1:
        raise ValueError("focus_mass must lie in (0.5, 1]")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")


def _resolve_device(
    features: Tensor,
    requested: str | torch.device,
) -> torch.device:
    if requested == "input":
        resolved = features.device
    elif requested == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return resolved


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)


def _initialize_student(model: _Student, generator: torch.Generator) -> None:
    with torch.no_grad():
        for layer in model.modules():
            if not isinstance(layer, nn.Linear):
                continue
            bound = 1 / math.sqrt(layer.in_features)
            layer.weight.uniform_(-bound, bound, generator=generator)
            if layer.bias is not None:
                layer.bias.uniform_(-bound, bound, generator=generator)


def _resolve_environment_ids(
    environment_ids: Tensor,
    count: int,
    device: torch.device,
) -> Tensor:
    if not isinstance(environment_ids, Tensor):
        raise TypeError("environment_ids must be a tensor")
    if environment_ids.shape != (count,):
        raise ValueError("environment_ids must contain one value per train example")
    supplied = environment_ids.detach().to(device=device)
    if supplied.is_floating_point() and (
        not torch.all(torch.isfinite(supplied))
        or not torch.all(supplied == supplied.round())
    ):
        raise ValueError("environment_ids must contain finite integers")
    resolved = supplied.to(dtype=torch.long)
    if torch.any(resolved < 0):
        raise ValueError("environment_ids must be non-negative")
    environment_count = int(resolved.max().item()) + 1
    if environment_count < 2:
        raise ValueError("at least two environments are required")
    counts = torch.bincount(resolved, minlength=environment_count)
    if torch.any(counts == 0):
        raise ValueError("environment_ids must be contiguous and non-empty")
    return resolved


def _resolve_trust(train_trust: Tensor | None, labels: Tensor) -> Tensor:
    if train_trust is None:
        return torch.ones_like(labels)
    if not isinstance(train_trust, Tensor):
        raise TypeError("train_trust must be a tensor")
    trust = train_trust.detach().to(device=labels.device, dtype=labels.dtype)
    if trust.shape != labels.shape:
        raise ValueError("train_trust must contain one value per train example")
    if not torch.all(torch.isfinite(trust)) or torch.any(trust < 0):
        raise ValueError("train_trust must be finite and non-negative")
    if trust.sum() <= 0:
        raise ValueError("train_trust must have positive total mass")
    return trust


def _standardization_and_centers(
    train_features: Tensor,
    environment_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    feature_mean = train_features.mean(dim=0, keepdim=True)
    centered = train_features - feature_mean
    feature_scale = centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    standardized = centered / feature_scale
    environment_count = int(environment_ids.max().item()) + 1
    centers = torch.stack(
        [
            standardized[environment_ids == environment].mean(dim=0)
            for environment in range(environment_count)
        ]
    )
    return feature_mean, feature_scale, centers


def _ordinary_probabilities(trust: Tensor) -> Tensor:
    return trust / trust.sum()


def _balanced_probabilities(environment_ids: Tensor, trust: Tensor) -> Tensor:
    environment_count = int(environment_ids.max().item()) + 1
    trust_mass = torch.zeros(
        environment_count,
        device=trust.device,
        dtype=trust.dtype,
    ).scatter_add_(0, environment_ids, trust)
    if torch.any(trust_mass <= 0):
        raise ValueError("every environment must have positive train_trust mass")
    probabilities = trust / trust_mass[environment_ids] / environment_count
    return probabilities / probabilities.sum()


def _focused_probabilities(
    environment_ids: Tensor,
    trust: Tensor,
    target_environment: int,
    focus_mass: float,
) -> Tensor:
    focused = environment_ids == target_environment
    focus_trust = trust[focused].sum()
    other_trust = trust[~focused].sum()
    if focus_trust <= 0:
        raise ValueError("every environment must have positive train_trust mass")
    if other_trust <= 0:
        return _ordinary_probabilities(trust)
    probabilities = torch.empty_like(trust)
    probabilities[focused] = focus_mass * trust[focused] / focus_trust
    probabilities[~focused] = (1 - focus_mass) * trust[~focused] / other_trust
    return probabilities / probabilities.sum()


def _train_student(
    model: _Student,
    train_features: Tensor,
    labels: Tensor,
    probabilities: Tensor,
    *,
    config: EnvironmentMixtureConfig,
    generator: torch.Generator,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    for _ in range(config.training_steps):
        batch_indices = torch.multinomial(
            probabilities,
            config.batch_size,
            replacement=True,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_features[batch_indices])
        loss = F.binary_cross_entropy_with_logits(logits, labels[batch_indices])
        loss.backward()
        optimizer.step()


def _prediction_correlation(probabilities: Tensor) -> tuple[Tensor, Tensor]:
    centered = probabilities - probabilities.mean(dim=1, keepdim=True)
    gram = centered @ centered.T
    norms = centered.square().sum(dim=1).sqrt()
    denominator = norms[:, None] * norms[None, :]
    correlation = torch.where(
        denominator > torch.finfo(probabilities.dtype).eps,
        gram / denominator.clamp_min(torch.finfo(probabilities.dtype).eps),
        torch.zeros_like(gram),
    ).clamp(-1, 1)
    correlation.fill_diagonal_(1)
    student_count = probabilities.shape[0]
    if student_count == 1:
        off_diagonal = correlation.new_tensor(float("nan"))
    else:
        off_diagonal = (correlation.sum() - correlation.diagonal().sum()) / (
            student_count * (student_count - 1)
        )
    return correlation, off_diagonal


def _ensemble_predictions(
    models: list[_Student],
    train_features: Tensor,
    labels: Tensor,
    test_features: Tensor,
    model_seeds: Tensor,
    compute: EnvironmentMixtureCompute,
) -> EnsemblePredictions:
    with torch.no_grad():
        train_logits = torch.stack([model(train_features) for model in models])
        test_logits = torch.stack([model(test_features) for model in models])
        train_probabilities = train_logits.sigmoid()
        test_probabilities = test_logits.sigmoid()
        errors = (
            ((train_logits >= 0) != labels.to(dtype=torch.bool).unsqueeze(0))
            .to(dtype=torch.float32)
            .mean(dim=1)
        )
        correlation, off_diagonal = _prediction_correlation(train_probabilities)
        mean_probabilities = test_probabilities.mean(dim=0)
        mean_logits = torch.logit(
            mean_probabilities,
            eps=torch.finfo(mean_probabilities.dtype).eps,
        )
    return EnsemblePredictions(
        logits=test_logits,
        probabilities=test_probabilities,
        mean_logits=mean_logits,
        mean_probabilities=mean_probabilities,
        model_seeds=model_seeds,
        diagnostics=StudentDiagnostics(
            error_rates=errors,
            prediction_correlation=correlation,
            mean_off_diagonal_correlation=off_diagonal,
        ),
        compute=compute,
    )


def train_environment_mixture(
    train_features: Tensor,
    observed_labels: Tensor,
    test_features: Tensor,
    environment_ids: Tensor,
    *,
    config: EnvironmentMixtureConfig | None = None,
    train_trust: Tensor | None = None,
    device: str | torch.device | None = None,
) -> EnvironmentMixtureResult:
    """Train equal-compute ordinary and environment-specialized ensembles.

    Args:
        train_features: Non-empty ``(train_example, feature)`` matrix.
        observed_labels: Binary labels visible to training.
        test_features: ``(test_example, feature)`` matrix to predict.
        environment_ids: Discovered, contiguous train-environment identifiers.
        config: Architecture, optimizer, budget, and deterministic seed settings.
        train_trust: Optional non-negative observed-data trust per train example.
        device: Optional execution-device override for ``config.device``.

    All tensor outputs stay on the selected execution device.
    """

    config = config or EnvironmentMixtureConfig()
    _validate_config(config)
    if not isinstance(train_features, Tensor) or not isinstance(test_features, Tensor):
        raise TypeError("train_features and test_features must be tensors")
    if not isinstance(observed_labels, Tensor):
        raise TypeError("observed_labels must be a tensor")
    if train_features.ndim != 2 or train_features.shape[0] == 0:
        raise ValueError("train_features must be a non-empty matrix")
    if test_features.ndim != 2:
        raise ValueError("test_features must be a matrix")
    if test_features.shape[1] != train_features.shape[1]:
        raise ValueError("train_features and test_features must have equal width")
    if observed_labels.shape != (train_features.shape[0],):
        raise ValueError("observed_labels must contain one scalar per train example")
    if not torch.all(torch.isfinite(train_features)):
        raise ValueError("train_features must be finite")
    if not torch.all(torch.isfinite(test_features)):
        raise ValueError("test_features must be finite")
    if not torch.all(torch.isfinite(observed_labels)):
        raise ValueError("observed_labels must be finite")
    if torch.any((observed_labels != 0) & (observed_labels != 1)):
        raise ValueError("observed_labels must be binary")

    selected_device = _resolve_device(
        train_features,
        config.device if device is None else device,
    )
    training = train_features.detach().to(
        device=selected_device,
        dtype=torch.float32,
    )
    testing = test_features.detach().to(device=selected_device, dtype=torch.float32)
    labels = observed_labels.detach().to(
        device=selected_device,
        dtype=torch.float32,
    )
    resolved_ids = _resolve_environment_ids(
        environment_ids,
        labels.numel(),
        selected_device,
    )
    trust = _resolve_trust(train_trust, labels)
    environment_count = int(resolved_ids.max().item()) + 1
    student_environment_ids = torch.arange(
        environment_count,
        device=selected_device,
        dtype=torch.long,
    )
    if config.include_balanced_student:
        student_environment_ids = torch.cat(
            (student_environment_ids, student_environment_ids.new_tensor([-1]))
        )
    student_count = student_environment_ids.numel()
    model_seeds = torch.tensor(
        [config.seed + 1_000_003 * index for index in range(student_count)],
        device=selected_device,
        dtype=torch.long,
    )

    ordinary_models: list[_Student] = []
    specialist_models: list[_Student] = []
    ordinary_sampling = _ordinary_probabilities(trust)
    balanced_sampling = _balanced_probabilities(resolved_ids, trust)
    for student_index, environment in enumerate(student_environment_ids.tolist()):
        seed = int(model_seeds[student_index].item())
        ordinary = _Student(
            training.shape[1],
            config.hidden_dimensions,
        ).to(selected_device)
        specialist = _Student(
            training.shape[1],
            config.hidden_dimensions,
        ).to(selected_device)
        _initialize_student(ordinary, _generator(selected_device, seed))
        _initialize_student(specialist, _generator(selected_device, seed))
        _train_student(
            ordinary,
            training,
            labels,
            ordinary_sampling,
            config=config,
            generator=_generator(selected_device, seed + 499),
        )
        specialist_sampling = (
            balanced_sampling
            if environment < 0
            else _focused_probabilities(
                resolved_ids,
                trust,
                environment,
                config.focus_mass,
            )
        )
        _train_student(
            specialist,
            training,
            labels,
            specialist_sampling,
            config=config,
            generator=_generator(selected_device, seed + 499),
        )
        ordinary_models.append(ordinary)
        specialist_models.append(specialist)

    compute = EnvironmentMixtureCompute(
        model_fits=student_count,
        optimizer_steps=student_count * config.training_steps,
        backward_examples=student_count * config.training_steps * config.batch_size,
        train_diagnostic_forward_examples=student_count * training.shape[0],
        test_forward_examples=student_count * testing.shape[0],
    )
    ordinary_output = _ensemble_predictions(
        ordinary_models,
        training,
        labels,
        testing,
        model_seeds,
        compute,
    )
    specialist_base = _ensemble_predictions(
        specialist_models,
        training,
        labels,
        testing,
        model_seeds,
        compute,
    )

    feature_mean, feature_scale, centers = _standardization_and_centers(
        training,
        resolved_ids,
    )
    standardized_test = (testing - feature_mean) / feature_scale
    routing_ids = torch.cdist(standardized_test, centers).argmin(dim=1)
    # The first ``environment_count`` students are ordered by environment ID.
    routed_student_indices = routing_ids
    test_indices = torch.arange(testing.shape[0], device=selected_device)
    routed_logits = specialist_base.logits[routed_student_indices, test_indices]
    routed_probabilities = specialist_base.probabilities[
        routed_student_indices,
        test_indices,
    ]
    specialist_output = SpecialistPredictions(
        logits=specialist_base.logits,
        probabilities=specialist_base.probabilities,
        mean_logits=specialist_base.mean_logits,
        mean_probabilities=specialist_base.mean_probabilities,
        model_seeds=specialist_base.model_seeds,
        diagnostics=specialist_base.diagnostics,
        compute=specialist_base.compute,
        student_environment_ids=student_environment_ids,
        routed_student_indices=routed_student_indices,
        routed_logits=routed_logits,
        routed_probabilities=routed_probabilities,
    )
    return EnvironmentMixtureResult(
        config=config,
        ordinary=ordinary_output,
        specialist=specialist_output,
        routing_environment_ids=routing_ids,
        environment_centers=centers,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )


# ``fit`` is a concise spelling for callers that treat this as a final estimator.
fit_environment_mixture = train_environment_mixture
