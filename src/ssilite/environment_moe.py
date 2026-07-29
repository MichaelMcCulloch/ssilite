"""Formal hard-routed MoE over label-free discovered environments.

The model in this module is one :class:`torch.nn.Module`, not a collection of
independently owned students.  Its expert parameters are contiguous tensors
with a leading expert axis, and its learned router executes exactly one expert
per example at inference.

Environment discovery remains outside this module.  Discovered environment IDs
act as pseudo-labels for the router and as teacher-forced dispatch targets
during training.  Teacher forcing is deliberate: it prevents an untrained
router from starving an expert before coherent specialization can emerge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class EnvironmentMoEConfig:
    """Architecture, optimizer, and deterministic training settings."""

    focus_mass: float = 0.9
    hidden_dimensions: int = 16
    training_steps: int = 80
    batch_size: int = 64
    learning_rate: float = 0.02
    weight_decay: float = 1e-4
    router_loss_weight: float = 1.0
    seed: int = 0
    device: str | torch.device = "input"


@dataclass(frozen=True)
class EnvironmentMoECompute:
    """Exact logical and physical work used by one trained MoE arm."""

    model_fits: int
    optimizer_steps: int
    expert_updates: int
    backward_examples: int
    router_training_examples: int
    train_diagnostic_forward_examples: int
    test_diagnostic_forward_examples: int
    sparse_inference_examples: int
    router_diagnostic_forward_examples: int


@dataclass(frozen=True)
class ExpertDiagnostics:
    """Observed-train errors and dependence between expert predictions."""

    error_rates: Tensor
    prediction_correlation: Tensor
    mean_off_diagonal_correlation: Tensor


@dataclass(frozen=True)
class MoEPredictions:
    """Dense diagnostics and genuine sparse top-1 predictions for one MoE."""

    logits: Tensor
    probabilities: Tensor
    mean_logits: Tensor
    mean_probabilities: Tensor
    router_logits: Tensor
    router_probabilities: Tensor
    routed_expert_indices: Tensor
    routed_logits: Tensor
    routed_probabilities: Tensor
    route_counts: Tensor
    router_train_accuracy: Tensor
    model_seed: Tensor
    diagnostics: ExpertDiagnostics
    compute: EnvironmentMoECompute


@dataclass(frozen=True)
class EnvironmentMoEResult:
    """Paired ordinary and environment-specialized formal MoEs."""

    config: EnvironmentMoEConfig
    device: torch.device
    ordinary: MoEPredictions
    specialist: MoEPredictions
    feature_mean: Tensor
    feature_scale: Tensor


class TensorizedExpertBank(nn.Module):
    """One-hidden-layer experts stored in four contiguous parameter tensors."""

    def __init__(
        self,
        *,
        num_experts: int,
        input_dimensions: int,
        hidden_dimensions: int,
    ) -> None:
        super().__init__()
        if min(num_experts, input_dimensions, hidden_dimensions) < 1:
            raise ValueError("expert count and dimensions must be positive")
        self.num_experts = num_experts
        self.input_dimensions = input_dimensions
        self.hidden_dimensions = hidden_dimensions
        self.input_weight = nn.Parameter(
            torch.empty(num_experts, hidden_dimensions, input_dimensions)
        )
        self.input_bias = nn.Parameter(torch.empty(num_experts, hidden_dimensions))
        self.output_weight = nn.Parameter(torch.empty(num_experts, hidden_dimensions))
        self.output_bias = nn.Parameter(torch.empty(num_experts))
        self.reset_parameters()

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        """Initialize every expert slice without relying on module lists."""

        input_bound = 1 / math.sqrt(self.input_dimensions)
        output_bound = 1 / math.sqrt(self.hidden_dimensions)
        with torch.no_grad():
            self.input_weight.uniform_(
                -input_bound,
                input_bound,
                generator=generator,
            )
            self.input_bias.uniform_(
                -input_bound,
                input_bound,
                generator=generator,
            )
            self.output_weight.uniform_(
                -output_bound,
                output_bound,
                generator=generator,
            )
            self.output_bias.uniform_(
                -output_bound,
                output_bound,
                generator=generator,
            )

    def _validate_features(self, features: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.input_dimensions:
            raise ValueError("features must have shape (example, input_dimensions)")

    def _validate_expert_ids(self, expert_ids: Tensor, count: int) -> Tensor:
        if expert_ids.shape != (count,):
            raise ValueError("expert_ids must contain one ID per example")
        resolved = expert_ids.to(device=self.input_weight.device, dtype=torch.long)
        if torch.any((resolved < 0) | (resolved >= self.num_experts)):
            raise ValueError("expert_ids are outside the expert bank")
        return resolved

    def all_logits(self, features: Tensor) -> Tensor:
        """Evaluate every expert for diagnostics, returning ``(expert, example)``."""

        self._validate_features(features)
        hidden = torch.tanh(
            torch.einsum("nd,ehd->enh", features, self.input_weight)
            + self.input_bias[:, None, :]
        )
        return (
            torch.einsum("enh,eh->en", hidden, self.output_weight)
            + self.output_bias[:, None]
        )

    def selected_logits(self, features: Tensor, expert_ids: Tensor) -> Tensor:
        """Evaluate only each example's selected expert."""

        self._validate_features(features)
        resolved_ids = self._validate_expert_ids(expert_ids, features.shape[0])
        selected_input_weight = self.input_weight[resolved_ids]
        hidden = torch.tanh(
            torch.bmm(selected_input_weight, features.unsqueeze(-1)).squeeze(-1)
            + self.input_bias[resolved_ids]
        )
        return (hidden * self.output_weight[resolved_ids]).sum(
            dim=-1
        ) + self.output_bias[resolved_ids]

    def expert_batch_logits(self, features: Tensor) -> Tensor:
        """Evaluate an expert-major ``(expert, batch, feature)`` tensor."""

        expected = (self.num_experts, self.input_dimensions)
        if (
            features.ndim != 3
            or features.shape[0] != expected[0]
            or features.shape[2] != expected[1]
        ):
            raise ValueError(
                "features must have shape (num_experts, batch, input_dimensions)"
            )
        hidden = torch.tanh(
            torch.einsum("ebd,ehd->ebh", features, self.input_weight)
            + self.input_bias[:, None, :]
        )
        return (
            torch.einsum("ebh,eh->eb", hidden, self.output_weight)
            + self.output_bias[:, None]
        )


class EnvironmentMoE(nn.Module):
    """A learned router plus a tensorized top-1 expert bank."""

    feature_mean: Tensor
    feature_scale: Tensor

    def __init__(
        self,
        *,
        input_dimensions: int,
        hidden_dimensions: int,
        num_experts: int,
        feature_mean: Tensor,
        feature_scale: Tensor,
    ) -> None:
        super().__init__()
        if feature_mean.numel() != input_dimensions:
            raise ValueError("feature_mean width must equal input_dimensions")
        if feature_scale.numel() != input_dimensions:
            raise ValueError("feature_scale width must equal input_dimensions")
        resolved_mean = feature_mean.detach().reshape(input_dimensions)
        resolved_scale = feature_scale.detach().reshape(input_dimensions)
        if not torch.all(torch.isfinite(resolved_mean)):
            raise ValueError("feature_mean must be finite")
        if not torch.all(torch.isfinite(resolved_scale)) or torch.any(
            resolved_scale <= 0
        ):
            raise ValueError("feature_scale must be finite and positive")
        self.input_dimensions = input_dimensions
        self.num_experts = num_experts
        self.experts = TensorizedExpertBank(
            num_experts=num_experts,
            input_dimensions=input_dimensions,
            hidden_dimensions=hidden_dimensions,
        )
        self.router = nn.Linear(input_dimensions, num_experts)
        self.register_buffer("feature_mean", resolved_mean.clone())
        self.register_buffer("feature_scale", resolved_scale.clone())

    def router_logits(self, features: Tensor) -> Tensor:
        """Return learned environment scores for each example."""

        if features.ndim != 2 or features.shape[1] != self.input_dimensions:
            raise ValueError("features must have shape (example, input_dimensions)")
        standardized = (features - self.feature_mean) / self.feature_scale
        return self.router(standardized)

    def route(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return router logits, probabilities, and hard top-1 IDs."""

        logits = self.router_logits(features)
        probabilities = logits.softmax(dim=-1)
        return logits, probabilities, logits.argmax(dim=-1)

    def all_expert_logits(self, features: Tensor) -> Tensor:
        """Evaluate all experts for the dense-mean causal ablation."""

        return self.experts.all_logits(features)

    def selected_expert_logits(
        self,
        features: Tensor,
        expert_ids: Tensor,
    ) -> Tensor:
        """Evaluate an explicit route plan without consulting the router."""

        return self.experts.selected_logits(features, expert_ids)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Execute learned top-1 sparse routing."""

        _, _, expert_ids = self.route(features)
        return self.selected_expert_logits(features, expert_ids), expert_ids


def _validate_config(config: EnvironmentMoEConfig) -> None:
    if min(config.hidden_dimensions, config.training_steps, config.batch_size) < 1:
        raise ValueError("model sizes, steps, and batch size must be positive")
    if not 0.5 < config.focus_mass <= 1:
        raise ValueError("focus_mass must lie in (0.5, 1]")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not math.isfinite(config.router_loss_weight) or config.router_loss_weight <= 0:
        raise ValueError("router_loss_weight must be positive")


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


def _standardization(train_features: Tensor) -> tuple[Tensor, Tensor]:
    feature_mean = train_features.mean(dim=0)
    centered = train_features - feature_mean
    feature_scale = centered.square().mean(dim=0).sqrt().clamp_min(1e-6)
    return feature_mean, feature_scale


def _initialize_model(
    model: EnvironmentMoE,
    generator: torch.Generator,
) -> None:
    model.experts.reset_parameters(generator)
    router_bound = 1 / math.sqrt(model.input_dimensions)
    with torch.no_grad():
        model.router.weight.uniform_(
            -router_bound,
            router_bound,
            generator=generator,
        )
        model.router.bias.uniform_(
            -router_bound,
            router_bound,
            generator=generator,
        )


def _train_model(
    model: EnvironmentMoE,
    train_features: Tensor,
    labels: Tensor,
    environment_ids: Tensor,
    trust: Tensor,
    *,
    specialist: bool,
    config: EnvironmentMoEConfig,
) -> None:
    environment_count = model.num_experts
    ordinary = _ordinary_probabilities(trust)
    if specialist:
        task_probabilities = torch.stack(
            [
                _focused_probabilities(
                    environment_ids,
                    trust,
                    environment,
                    config.focus_mass,
                )
                for environment in range(environment_count)
            ]
        )
    else:
        task_probabilities = ordinary.expand(environment_count, -1)
    router_probabilities = _balanced_probabilities(environment_ids, trust)
    task_generators = [
        _generator(train_features.device, config.seed + 499 + 1_000_003 * expert)
        for expert in range(environment_count)
    ]
    router_generator = _generator(train_features.device, config.seed + 7_919)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for _ in range(config.training_steps):
        task_indices = torch.stack(
            [
                torch.multinomial(
                    task_probabilities[expert],
                    config.batch_size,
                    replacement=True,
                    generator=task_generators[expert],
                )
                for expert in range(environment_count)
            ]
        )
        router_indices = torch.multinomial(
            router_probabilities,
            config.batch_size,
            replacement=True,
            generator=router_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        expert_logits = model.experts.expert_batch_logits(train_features[task_indices])
        task_losses = F.binary_cross_entropy_with_logits(
            expert_logits,
            labels[task_indices],
            reduction="none",
        )
        # Summing the per-expert means preserves the gradient magnitude each
        # expert would receive from its own optimizer.
        task_loss = task_losses.mean(dim=1).sum()
        router_loss = F.cross_entropy(
            model.router_logits(train_features[router_indices]),
            environment_ids[router_indices],
        )
        loss = task_loss + config.router_loss_weight * router_loss
        loss.backward()
        optimizer.step()


def _prediction_correlation(probabilities: Tensor) -> tuple[Tensor, Tensor]:
    centered = probabilities - probabilities.mean(dim=1, keepdim=True)
    gram = centered @ centered.T
    norms = centered.square().sum(dim=1).sqrt()
    denominator = norms[:, None] * norms[None, :]
    epsilon = torch.finfo(probabilities.dtype).eps
    correlation = torch.where(
        denominator > epsilon,
        gram / denominator.clamp_min(epsilon),
        torch.zeros_like(gram),
    ).clamp(-1, 1)
    correlation.fill_diagonal_(1)
    expert_count = probabilities.shape[0]
    off_diagonal = (correlation.sum() - correlation.diagonal().sum()) / (
        expert_count * (expert_count - 1)
    )
    return correlation, off_diagonal


def _predictions(
    model: EnvironmentMoE,
    train_features: Tensor,
    labels: Tensor,
    test_features: Tensor,
    environment_ids: Tensor,
    compute: EnvironmentMoECompute,
    seed: int,
) -> MoEPredictions:
    model.eval()
    with torch.no_grad():
        train_logits = model.all_expert_logits(train_features)
        test_logits = model.all_expert_logits(test_features)
        train_probabilities = train_logits.sigmoid()
        test_probabilities = test_logits.sigmoid()
        mean_probabilities = test_probabilities.mean(dim=0)
        mean_logits = torch.logit(
            mean_probabilities,
            eps=torch.finfo(mean_probabilities.dtype).eps,
        )
        router_logits, router_probabilities, routed_expert_indices = model.route(
            test_features
        )
        routed_logits = model.selected_expert_logits(
            test_features,
            routed_expert_indices,
        )
        routed_probabilities = routed_logits.sigmoid()
        route_counts = torch.bincount(
            routed_expert_indices,
            minlength=model.num_experts,
        )
        _, _, train_routes = model.route(train_features)
        router_train_accuracy = (
            (train_routes == environment_ids).to(dtype=torch.float32).mean()
        )
        errors = (
            ((train_logits >= 0) != labels.to(dtype=torch.bool).unsqueeze(0))
            .to(dtype=torch.float32)
            .mean(dim=1)
        )
        correlation, off_diagonal = _prediction_correlation(train_probabilities)
    return MoEPredictions(
        logits=test_logits,
        probabilities=test_probabilities,
        mean_logits=mean_logits,
        mean_probabilities=mean_probabilities,
        router_logits=router_logits,
        router_probabilities=router_probabilities,
        routed_expert_indices=routed_expert_indices,
        routed_logits=routed_logits,
        routed_probabilities=routed_probabilities,
        route_counts=route_counts,
        router_train_accuracy=router_train_accuracy,
        model_seed=torch.tensor(seed, device=test_features.device, dtype=torch.long),
        diagnostics=ExpertDiagnostics(
            error_rates=errors,
            prediction_correlation=correlation,
            mean_off_diagonal_correlation=off_diagonal,
        ),
        compute=compute,
    )


def train_environment_moe(
    train_features: Tensor,
    observed_labels: Tensor,
    test_features: Tensor,
    environment_ids: Tensor,
    *,
    config: EnvironmentMoEConfig | None = None,
    train_trust: Tensor | None = None,
    device: str | torch.device | None = None,
) -> EnvironmentMoEResult:
    """Train paired ordinary and environment-specialized formal MoEs.

    Training receives only observed features, observed binary labels, and
    label-free discovered environment IDs. All tensor outputs remain on the
    selected execution device.
    """

    config = config or EnvironmentMoEConfig()
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
    testing = test_features.detach().to(
        device=selected_device,
        dtype=torch.float32,
    )
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
    feature_mean, feature_scale = _standardization(training)

    def new_model() -> EnvironmentMoE:
        model = EnvironmentMoE(
            input_dimensions=training.shape[1],
            hidden_dimensions=config.hidden_dimensions,
            num_experts=environment_count,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        ).to(selected_device)
        _initialize_model(model, _generator(selected_device, config.seed))
        return model

    ordinary_model = new_model()
    specialist_model = new_model()
    _train_model(
        ordinary_model,
        training,
        labels,
        resolved_ids,
        trust,
        specialist=False,
        config=config,
    )
    _train_model(
        specialist_model,
        training,
        labels,
        resolved_ids,
        trust,
        specialist=True,
        config=config,
    )

    compute = EnvironmentMoECompute(
        model_fits=1,
        optimizer_steps=config.training_steps,
        expert_updates=environment_count * config.training_steps,
        backward_examples=(
            environment_count * config.training_steps * config.batch_size
        ),
        router_training_examples=config.training_steps * config.batch_size,
        train_diagnostic_forward_examples=environment_count * training.shape[0],
        test_diagnostic_forward_examples=environment_count * testing.shape[0],
        sparse_inference_examples=testing.shape[0],
        router_diagnostic_forward_examples=training.shape[0] + testing.shape[0],
    )
    ordinary = _predictions(
        ordinary_model,
        training,
        labels,
        testing,
        resolved_ids,
        compute,
        config.seed,
    )
    specialist = _predictions(
        specialist_model,
        training,
        labels,
        testing,
        resolved_ids,
        compute,
        config.seed,
    )
    if not torch.equal(
        ordinary.routed_expert_indices,
        specialist.routed_expert_indices,
    ):
        raise RuntimeError("paired routers diverged despite identical training")
    return EnvironmentMoEResult(
        config=config,
        device=selected_device,
        ordinary=ordinary,
        specialist=specialist,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )


fit_environment_moe = train_environment_moe
