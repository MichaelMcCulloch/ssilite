"""Expected/unexpected-uncertainty expert spawning.

This module is intentionally separate from :mod:`ssilite.environment_moe`.
The fixed-environment benchmark receives discovered environment IDs up front;
the model here begins with one active expert and must earn every later route
from observed features, observed labels, and held-out predictive evidence.

The controller is ordinary Python state rather than an ``nn.Module``.  It sees
detached pre-update losses, owns calibration/proposal state, and cannot receive
task gradients.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .environment_ensemble import discover_feature_environments

type SpawnerMode = Literal[
    "single",
    "raw_loss",
    "expected_only",
    "unvalidated",
    "joint",
]
type ExpertArchitecture = Literal["stable_latent", "plain"]
type RoutingStrategy = Literal["learned", "prototype"]
type ExpertParameters = tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class SpawningMoEConfig:
    """Architecture, evidence thresholds, budgets, and deterministic seeds."""

    max_experts: int = 3
    hidden_dimensions: int = 16
    latent_dimensions: int = 16
    expert_architecture: ExpertArchitecture = "stable_latent"
    routing_strategy: RoutingStrategy = "learned"
    batch_size: int = 32
    learning_rate: float = 0.02
    weight_decay: float = 1e-4
    warmup_count: int = 64
    calibration_capacity: int = 512
    surprise_tail_probability: float = 0.10
    raw_loss_threshold: float = 0.90
    proposal_interval: int = 64
    proposal_clusters: int = 3
    proposal_min_support: int = 32
    proposal_buffer_capacity: int = 512
    proposal_validation_fraction: float = 0.40
    kmeans_iterations: int = 20
    cooldown_examples: int = 64
    beta_prior_a: float = 1.0
    beta_prior_b: float = 1.0
    noise_epsilon: float = 0.01
    challenger_steps: int = 80
    challenger_learning_rate: float = 0.03
    challenger_anchor_weight: float = 0.5
    bootstrap_samples: int = 256
    confidence_alpha: float = 0.05
    practical_margin: float = 0.03
    router_steps: int = 60
    router_learning_rate: float = 0.05
    router_min_proposal_accuracy: float = 0.70
    router_min_anchor_accuracy: float = 0.85
    prototype_variance_floor: float = 0.05
    collateral_tolerance: float = 0.05
    collateral_min_support: int = 8
    collateral_practical_margin: float = 0.00
    context_hazard: float = 0.05
    context_persistence: float = 0.90
    birth_posterior_threshold: float = 0.90
    replay_capacity: int = 512
    seed: int = 0
    device: str | torch.device = "input"


@dataclass(frozen=True)
class SpawningCompute:
    """Exact example-level logical work for one spawning arm."""

    task_forward_examples: int
    task_backward_examples: int
    controller_scoring_forward_examples: int
    clustering_distance_evaluations: int
    candidate_fit_forward_examples: int
    candidate_fit_backward_examples: int
    candidate_scoring_forward_examples: int
    router_training_forward_examples: int
    router_training_backward_examples: int
    sparse_inference_examples: int

    @property
    def total_forward_examples(self) -> int:
        return (
            self.task_forward_examples
            + self.controller_scoring_forward_examples
            + self.candidate_fit_forward_examples
            + self.candidate_scoring_forward_examples
            + self.router_training_forward_examples
            + self.sparse_inference_examples
        )


@dataclass(frozen=True)
class BirthEvidence:
    """Held-out and routing evidence recorded for one challenger."""

    parent_expert: int
    fit_support: int
    validation_support: int
    estimated_noise_rate: float
    mean_improvement: float
    lower_confidence_bound: float
    fit_improvement: float
    validation_has_both_classes: bool
    proposal_route_accuracy: float
    anchor_route_accuracy: float
    collateral_loss_change: float
    collateral_support: int
    collateral_mean_improvement: float
    collateral_lower_confidence_bound: float
    rule_log_bayes_factor: float
    unexpected_uncertainty: float
    context_switch_log_margin: float


@dataclass(frozen=True)
class BirthRecord:
    """Accepted expert lineage and proposal provenance."""

    proposal_id: int
    expert_id: int
    parent_expert: int
    activation_example: int
    member_example_ids: tuple[int, ...]
    evidence: BirthEvidence


@dataclass(frozen=True)
class RejectionRecord:
    """A resolved proposal that did not activate capacity."""

    proposal_id: int
    resolution_example: int
    reason: str
    member_example_ids: tuple[int, ...]
    evidence: BirthEvidence | None


@dataclass(frozen=True)
class Proposal:
    """A feature-coherent unresolved set with a permanent fit/validation split."""

    proposal_id: int
    created_at: int
    example_ids: Tensor
    features: Tensor
    labels: Tensor
    losses: Tensor
    fit_indices: Tensor
    validation_indices: Tensor


@dataclass(frozen=True)
class ChallengerDecision:
    """Provisional state and the reason it may or may not be activated."""

    accepted: bool
    reason: str
    evidence: BirthEvidence | None
    expert_parameters: ExpertParameters | None
    router_weight: Tensor | None
    router_bias: Tensor | None


@dataclass(frozen=True)
class SpawningPredictions:
    """Sparse test predictions and active-route diagnostics."""

    routed_logits: Tensor
    routed_probabilities: Tensor
    routed_expert_indices: Tensor
    route_counts: Tensor
    active_expert_mask: Tensor


@dataclass(frozen=True)
class SpawningTrainingResult:
    """One observed-data-only spawning arm."""

    mode: SpawnerMode
    config: SpawningMoEConfig
    device: torch.device
    predictions: SpawningPredictions
    births: tuple[BirthRecord, ...]
    rejections: tuple[RejectionRecord, ...]
    calibration_counts: tuple[int, ...]
    expected_uncertainties: tuple[float, ...]
    surprising_examples: int
    proposal_count: int
    unresolved_count: int
    compute: SpawningCompute
    diagnostics_json: str


@dataclass
class _WorkCounter:
    task_forward_examples: int = 0
    task_backward_examples: int = 0
    controller_scoring_forward_examples: int = 0
    clustering_distance_evaluations: int = 0
    candidate_fit_forward_examples: int = 0
    candidate_fit_backward_examples: int = 0
    candidate_scoring_forward_examples: int = 0
    router_training_forward_examples: int = 0
    router_training_backward_examples: int = 0
    sparse_inference_examples: int = 0

    def freeze(self) -> SpawningCompute:
        return SpawningCompute(**asdict(self))


class TensorizedLatentExpertBank(nn.Module):
    """Contiguous routed transforms operating in a compact latent space."""

    def __init__(
        self,
        *,
        num_experts: int,
        input_dimensions: int,
        latent_dimensions: int,
    ) -> None:
        super().__init__()
        if min(num_experts, input_dimensions, latent_dimensions) < 1:
            raise ValueError("expert count and dimensions must be positive")
        self.num_experts = num_experts
        self.input_dimensions = input_dimensions
        self.latent_dimensions = latent_dimensions
        shape = (num_experts, latent_dimensions, input_dimensions)
        self.input_weight = nn.Parameter(torch.empty(shape))
        self.input_bias = nn.Parameter(torch.empty(num_experts, latent_dimensions))
        self.output_weight = nn.Parameter(torch.empty(num_experts, latent_dimensions))
        self.output_bias = nn.Parameter(torch.empty(num_experts))
        self.reset_parameters()

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        input_bound = 1 / math.sqrt(self.input_dimensions)
        output_bound = 1 / math.sqrt(self.latent_dimensions)
        with torch.no_grad():
            for parameter in (self.input_weight, self.input_bias):
                parameter.uniform_(-input_bound, input_bound, generator=generator)
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

    def active_latents(
        self,
        features: Tensor,
        *,
        active_count: int,
    ) -> Tensor:
        return torch.tanh(
            torch.einsum("nd,ehd->enh", features, self.input_weight[:active_count])
            + self.input_bias[:active_count, None, :]
        )

    def selected_latents(
        self,
        features: Tensor,
        expert_ids: Tensor,
    ) -> Tensor:
        selected_input = self.input_weight[expert_ids]
        return torch.tanh(
            torch.bmm(selected_input, features.unsqueeze(-1)).squeeze(-1)
            + self.input_bias[expert_ids]
        )


class SpawningMoE(nn.Module):
    """Preallocated experts with an optional Stable-Latent-style vessel."""

    feature_mean: Tensor
    feature_scale: Tensor
    active_expert_mask: Tensor

    def __init__(
        self,
        *,
        input_dimensions: int,
        hidden_dimensions: int,
        latent_dimensions: int | None = None,
        expert_architecture: ExpertArchitecture = "stable_latent",
        max_experts: int,
        feature_mean: Tensor,
        feature_scale: Tensor,
    ) -> None:
        super().__init__()
        if expert_architecture not in {"stable_latent", "plain"}:
            raise ValueError(f"unknown expert architecture: {expert_architecture!r}")
        resolved_latent = (
            hidden_dimensions
            if expert_architecture == "plain"
            else (
                min(input_dimensions, hidden_dimensions)
                if latent_dimensions is None
                else latent_dimensions
            )
        )
        if min(input_dimensions, hidden_dimensions, resolved_latent, max_experts) < 1:
            raise ValueError("dimensions and maximum capacity must be positive")
        if feature_mean.shape != (input_dimensions,):
            raise ValueError("feature_mean must match input_dimensions")
        if feature_scale.shape != (input_dimensions,):
            raise ValueError("feature_scale must match input_dimensions")
        if not torch.all(torch.isfinite(feature_mean)):
            raise ValueError("feature_mean must be finite")
        if not torch.all(torch.isfinite(feature_scale)) or torch.any(
            feature_scale <= 0
        ):
            raise ValueError("feature_scale must be finite and positive")
        self.input_dimensions = input_dimensions
        self.hidden_dimensions = hidden_dimensions
        self.latent_dimensions = resolved_latent
        self.expert_architecture = expert_architecture
        self.max_experts = max_experts
        # Keep the always-on path deliberately linear. A full SiTU shared MLP
        # can itself learn the context-conditioned rule and erase the causal
        # distinction between scale-and-pray and earned specialist capacity.
        self.shared_output = (
            nn.Linear(input_dimensions, 1)
            if expert_architecture == "stable_latent"
            else None
        )
        self.experts = TensorizedLatentExpertBank(
            num_experts=max_experts,
            input_dimensions=input_dimensions,
            latent_dimensions=resolved_latent,
        )
        self.routed_norm = (
            nn.RMSNorm(resolved_latent)
            if expert_architecture == "stable_latent"
            else None
        )
        self.router = nn.Linear(input_dimensions, max_experts)
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        mask = torch.zeros(max_experts, dtype=torch.bool, device=feature_mean.device)
        mask[0] = True
        self.register_buffer("active_expert_mask", mask)

    @property
    def active_expert_count(self) -> int:
        return int(self.active_expert_mask.sum().item())

    def initialize(self, generator: torch.Generator) -> None:
        """Initialize expert zero and leave every dormant slice exactly zero."""

        self.experts.reset_parameters(generator)
        with torch.no_grad():
            layers = [self.router]
            if self.shared_output is not None:
                layers.append(self.shared_output)
            for layer in layers:
                bound = 1 / math.sqrt(layer.in_features)
                layer.weight.uniform_(-bound, bound, generator=generator)
                if layer.bias is not None:
                    layer.bias.uniform_(-bound, bound, generator=generator)
            if self.routed_norm is not None:
                self.routed_norm.weight.fill_(1)
            if self.max_experts > 1:
                self.experts.input_weight[1:].zero_()
                self.experts.input_bias[1:].zero_()
                self.experts.output_weight[1:].zero_()
                self.experts.output_bias[1:].zero_()
                self.router.weight[1:].zero_()
                self.router.bias[1:].zero_()

    def _validate_features(self, features: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.input_dimensions:
            raise ValueError("features must have shape (example, input_dimensions)")

    def active_expert_logits(self, features: Tensor) -> Tensor:
        """Evaluate active experts only, returning ``(active, example)``."""

        self._validate_features(features)
        active = self.active_expert_count
        routed = self.experts.active_latents(
            features,
            active_count=active,
        )
        if self.expert_architecture == "plain":
            return (routed * self.experts.output_weight[:active, None, :]).sum(
                dim=-1
            ) + self.experts.output_bias[:active, None]
        if self.routed_norm is None:
            raise RuntimeError("stable-latent experts require RMS normalization")
        routed_logits = (
            self.routed_norm(routed) * self.experts.output_weight[:active, None, :]
        ).sum(dim=-1) + self.experts.output_bias[:active, None]
        return routed_logits + self.shared_logits(features).unsqueeze(0)

    def shared_logits(self, features: Tensor) -> Tensor:
        self._validate_features(features)
        if self.shared_output is None:
            return torch.zeros(
                features.shape[0],
                device=features.device,
                dtype=features.dtype,
            )
        return self.shared_output(features).squeeze(-1)

    def active_router_logits(self, features: Tensor) -> Tensor:
        """Compute router outputs only for active parameter rows."""

        self._validate_features(features)
        standardized = (features - self.feature_mean) / self.feature_scale
        active = self.active_expert_count
        return F.linear(
            standardized,
            self.router.weight[:active],
            self.router.bias[:active],
        )

    def route(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.active_router_logits(features)
        probabilities = logits.softmax(dim=-1)
        active_ids = torch.arange(
            self.active_expert_count,
            device=features.device,
            dtype=torch.long,
        )
        return logits, probabilities, active_ids[logits.argmax(dim=-1)]

    def selected_expert_logits(self, features: Tensor, expert_ids: Tensor) -> Tensor:
        self._validate_features(features)
        if expert_ids.shape != (features.shape[0],):
            raise ValueError("expert_ids must contain one ID per example")
        if torch.any((expert_ids < 0) | (expert_ids >= self.active_expert_count)):
            raise ValueError("expert_ids must select active experts")
        resolved_ids = expert_ids.to(device=features.device, dtype=torch.long)
        routed = self.experts.selected_latents(
            features,
            resolved_ids,
        )
        if self.expert_architecture == "plain":
            return (routed * self.experts.output_weight[resolved_ids]).sum(
                dim=-1
            ) + self.experts.output_bias[resolved_ids]
        if self.routed_norm is None:
            raise RuntimeError("stable-latent experts require RMS normalization")
        routed_logits = (
            self.routed_norm(routed) * self.experts.output_weight[resolved_ids]
        ).sum(dim=-1) + self.experts.output_bias[resolved_ids]
        return self.shared_logits(features) + routed_logits

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        _, _, expert_ids = self.route(features)
        return self.selected_expert_logits(features, expert_ids), expert_ids

    def parent_parameters(
        self,
        expert_id: int,
    ) -> ExpertParameters:
        if not 0 <= expert_id < self.active_expert_count:
            raise ValueError("parent expert must be active")
        return (
            self.experts.input_weight[expert_id].detach().clone(),
            self.experts.input_bias[expert_id].detach().clone(),
            self.experts.output_weight[expert_id].detach().clone(),
            self.experts.output_bias[expert_id].detach().clone(),
        )

    def activate(
        self,
        *,
        expert_parameters: ExpertParameters,
        router_weight: Tensor,
        router_bias: Tensor,
    ) -> int:
        """Copy accepted provisional state into the next dormant slice."""

        expert_id = self.active_expert_count
        if expert_id >= self.max_experts:
            raise RuntimeError("expert capacity is exhausted")
        if router_weight.shape != (expert_id + 1, self.input_dimensions):
            raise ValueError("router_weight must cover old experts plus the birth")
        if router_bias.shape != (expert_id + 1,):
            raise ValueError("router_bias must cover old experts plus the birth")
        targets = (
            self.experts.input_weight[expert_id],
            self.experts.input_bias[expert_id],
            self.experts.output_weight[expert_id],
            self.experts.output_bias[expert_id],
        )
        with torch.no_grad():
            for target, source in zip(targets, expert_parameters, strict=True):
                if target.shape != source.shape:
                    raise ValueError("provisional expert parameter shape mismatch")
                target.copy_(source)
            self.router.weight[: expert_id + 1].copy_(router_weight)
            self.router.bias[: expert_id + 1].copy_(router_bias)
            self.active_expert_mask[expert_id] = True
        return expert_id


class EmpiricalCalibration:
    """Bounded expert-local empirical prequential loss distributions."""

    def __init__(
        self,
        *,
        max_experts: int,
        capacity: int,
        warmup_count: int,
    ) -> None:
        if min(max_experts, capacity, warmup_count) < 1:
            raise ValueError("calibration sizes must be positive")
        self.capacity = capacity
        self.warmup_count = warmup_count
        self._histories = [deque[float](maxlen=capacity) for _ in range(max_experts)]

    def count(self, expert_id: int) -> int:
        return len(self._histories[expert_id])

    def counts(self) -> tuple[int, ...]:
        return tuple(len(history) for history in self._histories)

    def add(self, expert_id: int, loss: float) -> None:
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("calibration losses must be finite and non-negative")
        self._histories[expert_id].append(loss)

    def tail_probability(self, expert_id: int, loss: float) -> float:
        """Return the smoothed empirical upper-tail probability."""

        history = self._histories[expert_id]
        if len(history) < self.warmup_count:
            return 1.0
        tail_count = sum(previous >= loss for previous in history)
        return (1 + tail_count) / (len(history) + 1)

    def expected_uncertainty(
        self,
        expert_id: int,
        *,
        prior_error: float = 1.0,
        prior_correct: float = 1.0,
    ) -> float:
        """Beta-Bernoulli estimate of within-context cue invalidity.

        Binary cross-entropy exceeds ``log(2)`` exactly when the expert assigns
        less than half its probability to the observed label.
        """

        history = self._histories[expert_id]
        errors = sum(loss > math.log(2) for loss in history)
        return (prior_error + errors) / (prior_error + prior_correct + len(history))


@dataclass(frozen=True)
class _UnresolvedExample:
    example_id: int
    feature: Tensor
    label: Tensor
    losses: Tensor


class UnexpectedUncertaintyController:
    """Detached calibration, unresolved-surprise, and proposal state."""

    def __init__(self, config: SpawningMoEConfig, mode: SpawnerMode) -> None:
        self.config = config
        self.mode = mode
        self.calibration = EmpiricalCalibration(
            max_experts=config.max_experts,
            capacity=config.calibration_capacity,
            warmup_count=config.warmup_count,
        )
        self.observation_count = 0
        self.next_proposal_id = 0
        self.surprise_count = 0
        self.cooldown_until = 0
        self._unresolved: deque[_UnresolvedExample] = deque()
        self.births: list[BirthRecord] = []
        self.rejections: list[RejectionRecord] = []

    @property
    def unresolved_count(self) -> int:
        return len(self._unresolved)

    def _admit_losses(self, losses: Tensor) -> None:
        for expert_id, loss in enumerate(losses.tolist()):
            self.calibration.add(expert_id, float(loss))

    def _is_surprising(self, losses: Tensor) -> bool:
        if self.mode == "raw_loss":
            return bool(float(losses.min().item()) >= self.config.raw_loss_threshold)
        if self.mode in {"expected_only", "unvalidated", "joint"}:
            tails = [
                self.calibration.tail_probability(expert_id, float(loss))
                for expert_id, loss in enumerate(losses.tolist())
            ]
            return all(tail <= self.config.surprise_tail_probability for tail in tails)
        return False

    def observe_batch(
        self,
        *,
        example_ids: Tensor,
        features: Tensor,
        labels: Tensor,
        active_losses: Tensor,
        work: _WorkCounter | None = None,
    ) -> tuple[Proposal, ...]:
        """Consume detached pre-update losses and return proposals due now."""

        count = labels.numel()
        if example_ids.shape != (count,):
            raise ValueError("example_ids must contain one ID per example")
        if features.shape[0] != count or labels.shape != (count,):
            raise ValueError("features and labels must share the example axis")
        if active_losses.ndim != 2 or active_losses.shape[1] != count:
            raise ValueError("active_losses must have shape (active, example)")
        detached_losses = active_losses.detach().to(device="cpu")
        detached_features = features.detach().to(device="cpu")
        detached_labels = labels.detach().to(device="cpu")
        detached_ids = example_ids.detach().to(device="cpu", dtype=torch.long)

        for index in range(count):
            losses = detached_losses[:, index]
            self.observation_count += 1
            if self._is_surprising(losses):
                self.surprise_count += 1
                self._unresolved.append(
                    _UnresolvedExample(
                        example_id=int(detached_ids[index].item()),
                        feature=detached_features[index].clone(),
                        label=detached_labels[index].clone(),
                        losses=losses.clone(),
                    )
                )
                if len(self._unresolved) > self.config.proposal_buffer_capacity:
                    evicted = self._unresolved.popleft()
                    self._admit_losses(evicted.losses)
                    self.rejections.append(
                        RejectionRecord(
                            proposal_id=-1,
                            resolution_example=self.observation_count,
                            reason="buffer_eviction",
                            member_example_ids=(evicted.example_id,),
                            evidence=None,
                        )
                    )
            else:
                self._admit_losses(losses)

        if (
            self.mode in {"raw_loss", "unvalidated", "joint"}
            and self.observation_count >= self.config.proposal_interval
            and self.observation_count % self.config.proposal_interval < count
            and self.observation_count >= self.cooldown_until
        ):
            return self._form_proposals(work)
        return ()

    def _form_proposals(
        self,
        work: _WorkCounter | None,
    ) -> tuple[Proposal, ...]:
        if len(self._unresolved) < self.config.proposal_min_support:
            return ()
        records = list(self._unresolved)
        features = torch.stack([record.feature for record in records])
        cluster_count = min(self.config.proposal_clusters, len(records))
        if cluster_count == 1:
            assignments = torch.zeros(len(records), dtype=torch.long)
        else:
            assignments = discover_feature_environments(
                features,
                num_environments=cluster_count,
                iterations=self.config.kmeans_iterations,
                seed=self.config.seed + 31_337 + self.next_proposal_id,
                device="cpu",
            )
        if work is not None:
            work.clustering_distance_evaluations += (
                len(records) * cluster_count * (self.config.kmeans_iterations + 1)
            )

        proposals: list[Proposal] = []
        proposed_record_ids: set[int] = set()
        for cluster in range(cluster_count):
            positions = torch.nonzero(assignments == cluster).flatten()
            if positions.numel() < self.config.proposal_min_support:
                continue
            proposal_records = [records[int(position)] for position in positions]
            proposal_id = self.next_proposal_id
            self.next_proposal_id += 1
            generator = torch.Generator().manual_seed(
                self.config.seed + 104_729 * (proposal_id + 1)
            )
            permutation = torch.randperm(len(proposal_records), generator=generator)
            validation_count = max(
                1,
                round(len(proposal_records) * self.config.proposal_validation_fraction),
            )
            validation_count = min(validation_count, len(proposal_records) - 1)
            validation_indices = permutation[:validation_count]
            fit_indices = permutation[validation_count:]
            proposals.append(
                Proposal(
                    proposal_id=proposal_id,
                    created_at=self.observation_count,
                    example_ids=torch.tensor(
                        [record.example_id for record in proposal_records],
                        dtype=torch.long,
                    ),
                    features=torch.stack(
                        [record.feature for record in proposal_records]
                    ),
                    labels=torch.stack([record.label for record in proposal_records]),
                    losses=torch.stack(
                        [record.losses for record in proposal_records],
                        dim=1,
                    ),
                    fit_indices=fit_indices,
                    validation_indices=validation_indices,
                )
            )
            proposed_record_ids.update(id(record) for record in proposal_records)
        self._unresolved = deque(
            record for record in records if id(record) not in proposed_record_ids
        )
        return tuple(proposals)

    def resolve(
        self,
        proposal: Proposal,
        decision: ChallengerDecision,
        *,
        expert_id: int | None = None,
    ) -> None:
        member_ids = tuple(int(value) for value in proposal.example_ids.tolist())
        if decision.accepted:
            if expert_id is None or decision.evidence is None:
                raise ValueError("accepted proposals require an expert and evidence")
            self.births.append(
                BirthRecord(
                    proposal_id=proposal.proposal_id,
                    expert_id=expert_id,
                    parent_expert=decision.evidence.parent_expert,
                    activation_example=self.observation_count,
                    member_example_ids=member_ids,
                    evidence=decision.evidence,
                )
            )
            # Surprise was defined against the pre-birth active set. Remaining
            # records cannot be combined with later all-active losses without
            # silently changing the proposal statistic's dimensionality.
            for unresolved in self._unresolved:
                self._admit_losses(unresolved.losses)
            self._unresolved.clear()
            return

        for position in range(proposal.losses.shape[1]):
            self._admit_losses(proposal.losses[:, position])
        self.cooldown_until = self.observation_count + self.config.cooldown_examples
        self.rejections.append(
            RejectionRecord(
                proposal_id=proposal.proposal_id,
                resolution_example=self.observation_count,
                reason=decision.reason,
                member_example_ids=member_ids,
                evidence=decision.evidence,
            )
        )

    def diagnostics_json(self) -> str:
        """Serialize stable controller diagnostics without tensor objects."""

        payload = {
            "births": [asdict(record) for record in self.births],
            "calibration_counts": self.calibration.counts(),
            "cooldown_until": self.cooldown_until,
            "mode": self.mode,
            "observation_count": self.observation_count,
            "expected_uncertainties": tuple(
                self.calibration.expected_uncertainty(expert)
                for expert in range(self.config.max_experts)
            ),
            "rejections": [asdict(record) for record in self.rejections],
            "surprise_count": self.surprise_count,
            "unresolved_count": self.unresolved_count,
        }
        return json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class _ProvisionalExpert(nn.Module):
    shared_output_weight: Tensor
    shared_output_bias: Tensor
    routed_norm_weight: Tensor

    def __init__(
        self,
        model: SpawningMoE,
        parameters: ExpertParameters,
    ) -> None:
        super().__init__()
        (
            input_weight,
            input_bias,
            output_weight,
            output_bias,
        ) = parameters
        self.input_weight = nn.Parameter(input_weight)
        self.input_bias = nn.Parameter(input_bias)
        self.output_weight = nn.Parameter(output_weight)
        self.output_bias = nn.Parameter(output_bias)
        self.expert_architecture = model.expert_architecture
        if model.expert_architecture == "plain":
            return
        if model.routed_norm is None or model.shared_output is None:
            raise RuntimeError("stable-latent experts require shared and RMS modules")
        self.rms_epsilon = model.routed_norm.eps
        for name, tensor in (
            ("shared_output_weight", model.shared_output.weight),
            ("shared_output_bias", model.shared_output.bias),
            ("routed_norm_weight", model.routed_norm.weight),
        ):
            if tensor is None:
                raise RuntimeError(f"{name} unexpectedly has no tensor")
            self.register_buffer(name, tensor.detach().clone())

    def forward(self, features: Tensor) -> Tensor:
        routed = torch.tanh(F.linear(features, self.input_weight, self.input_bias))
        if self.expert_architecture == "plain":
            return F.linear(
                routed,
                self.output_weight.unsqueeze(0),
                self.output_bias.unsqueeze(0),
            ).squeeze(-1)
        shared_logits = F.linear(
            features,
            self.shared_output_weight,
            self.shared_output_bias,
        ).squeeze(-1)
        normalized = F.rms_norm(
            routed,
            (routed.shape[-1],),
            self.routed_norm_weight,
            self.rms_epsilon,
        )
        return shared_logits + F.linear(
            normalized,
            self.output_weight.unsqueeze(0),
            self.output_bias.unsqueeze(0),
        ).squeeze(-1)

    def detached_parameters(self) -> ExpertParameters:
        return (
            self.input_weight.detach().clone(),
            self.input_bias.detach().clone(),
            self.output_weight.detach().clone(),
            self.output_bias.detach().clone(),
        )


def _balanced_anchor_weights(routes: Tensor, active_count: int) -> Tensor:
    if routes.numel() == 0:
        return torch.empty_like(routes, dtype=torch.float32)
    counts = torch.bincount(routes, minlength=active_count).clamp_min(1)
    weights = counts[routes].reciprocal().to(dtype=torch.float32)
    return weights / weights.sum()


def _split_replay_anchors(
    routes: Tensor,
    *,
    active_count: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Make route-stratified challenger-fit and collateral-holdout anchors."""

    fit_parts: list[Tensor] = []
    validation_parts: list[Tensor] = []
    for route in range(active_count):
        indices = torch.nonzero(routes == route).flatten()
        if indices.numel() == 0:
            continue
        permutation = indices[
            torch.randperm(
                indices.numel(),
                device=indices.device,
                generator=generator,
            )
        ]
        fit_count = max(1, (indices.numel() + 1) // 2)
        fit_parts.append(permutation[:fit_count])
        if fit_count < indices.numel():
            validation_parts.append(permutation[fit_count:])
    fit = (
        torch.cat(fit_parts)
        if fit_parts
        else torch.empty(0, device=routes.device, dtype=torch.long)
    )
    validation = (
        torch.cat(validation_parts)
        if validation_parts
        else torch.empty(0, device=routes.device, dtype=torch.long)
    )
    return fit, validation


def _train_router(
    *,
    model: SpawningMoE,
    proposal_features: Tensor,
    anchor_features: Tensor,
    anchor_routes: Tensor,
    config: SpawningMoEConfig,
    seed: int,
    work: _WorkCounter,
) -> nn.Linear:
    active_count = model.active_expert_count
    new_expert = active_count
    router = nn.Linear(model.input_dimensions, active_count + 1).to(
        proposal_features.device
    )
    generator = _generator(proposal_features.device, seed)
    bound = 1 / math.sqrt(model.input_dimensions)
    with torch.no_grad():
        router.weight[:active_count].copy_(model.router.weight[:active_count])
        router.bias[:active_count].copy_(model.router.bias[:active_count])
        router.weight[new_expert].uniform_(-bound, bound, generator=generator)
        router.bias[new_expert].uniform_(-bound, bound, generator=generator)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=config.router_learning_rate,
        weight_decay=0,
    )
    standardized_proposal = (
        proposal_features - model.feature_mean
    ) / model.feature_scale
    standardized_anchors = (anchor_features - model.feature_mean) / model.feature_scale
    proposal_targets = torch.full(
        (proposal_features.shape[0],),
        new_expert,
        device=proposal_features.device,
        dtype=torch.long,
    )
    anchor_weights = _balanced_anchor_weights(anchor_routes, active_count).to(
        proposal_features.device
    )
    for _ in range(config.router_steps):
        optimizer.zero_grad(set_to_none=True)
        proposal_loss = F.cross_entropy(
            router(standardized_proposal),
            proposal_targets,
        )
        if anchor_features.shape[0]:
            anchor_losses = F.cross_entropy(
                router(standardized_anchors),
                anchor_routes,
                reduction="none",
            )
            loss = proposal_loss + (anchor_losses * anchor_weights).sum()
        else:
            loss = proposal_loss
        loss.backward()
        optimizer.step()
    examples_per_step = proposal_features.shape[0] + anchor_features.shape[0]
    work.router_training_forward_examples += config.router_steps * examples_per_step
    work.router_training_backward_examples += config.router_steps * examples_per_step
    return router


def _prototype_router(
    *,
    model: SpawningMoE,
    proposal_features: Tensor,
    anchor_features: Tensor,
    anchor_routes: Tensor,
    variance_floor: float,
) -> nn.Linear:
    """Build an equal-prior diagonal-LDA gate with no learned parameters."""

    active_count = model.active_expert_count
    standardized_proposal = (
        proposal_features - model.feature_mean
    ) / model.feature_scale
    standardized_anchors = (anchor_features - model.feature_mean) / model.feature_scale
    groups: list[Tensor] = []
    centroids: list[Tensor] = []
    for expert_id in range(active_count):
        members = standardized_anchors[anchor_routes == expert_id]
        if members.shape[0]:
            groups.append(members)
            centroids.append(members.mean(dim=0))
        else:
            # A 512-point replay normally represents every active regime.
            # Zero is the conservative fallback when an expert is absent:
            # router weights do not retain enough information to reconstruct
            # its centroid after the previous precision scaling.
            centroids.append(torch.zeros_like(standardized_proposal[0]))
            groups.append(centroids[-1].unsqueeze(0))
    groups.append(standardized_proposal)
    centroids.append(standardized_proposal.mean(dim=0))
    stacked = torch.stack(centroids)
    residual_sum = torch.zeros_like(stacked[0])
    for group, centroid in zip(groups, centroids, strict=True):
        residual_sum += (group - centroid.unsqueeze(0)).square().sum(dim=0)
    residual_degrees = max(1, sum(max(0, group.shape[0] - 1) for group in groups))
    precision = (residual_sum / residual_degrees).clamp_min(variance_floor).reciprocal()
    router = nn.Linear(model.input_dimensions, active_count + 1).to(
        proposal_features.device
    )
    with torch.no_grad():
        router.weight.copy_(2 * stacked * precision)
        router.bias.copy_(-(stacked.square() * precision).sum(dim=-1))
    return router


def _provisional_router(
    *,
    model: SpawningMoE,
    proposal_features: Tensor,
    anchor_features: Tensor,
    anchor_routes: Tensor,
    config: SpawningMoEConfig,
    seed: int,
    work: _WorkCounter,
) -> nn.Linear:
    if config.routing_strategy == "prototype":
        return _prototype_router(
            model=model,
            proposal_features=proposal_features,
            anchor_features=anchor_features,
            anchor_routes=anchor_routes,
            variance_floor=config.prototype_variance_floor,
        )
    return _train_router(
        model=model,
        proposal_features=proposal_features,
        anchor_features=anchor_features,
        anchor_routes=anchor_routes,
        config=config,
        seed=seed,
        work=work,
    )


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)


def _bootstrap_lower_bound(
    improvements: Tensor,
    *,
    samples: int,
    alpha: float,
    generator: torch.Generator,
) -> float:
    count = improvements.numel()
    bootstrap_indices = torch.randint(
        count,
        (samples, count),
        device=improvements.device,
        generator=generator,
    )
    means = improvements[bootstrap_indices].mean(dim=1).sort().values
    position = min(samples - 1, max(0, math.floor(alpha * samples)))
    return float(means[position].item())


def evaluate_birth_proposal(
    model: SpawningMoE,
    proposal: Proposal,
    replay_features: Tensor,
    replay_labels: Tensor,
    replay_routes: Tensor,
    *,
    mode: Literal["raw_loss", "unvalidated", "joint"],
    config: SpawningMoEConfig,
    work: _WorkCounter | None = None,
) -> ChallengerDecision:
    """Fit and score one challenger using only observed tensors.

    The proposal's validation indices never enter challenger or provisional
    router optimization.  ``raw_loss`` deliberately ignores predictive model
    comparison; ``unvalidated`` uses in-sample fit improvement; ``joint`` uses
    paired untouched validation and a seeded bootstrap lower bound.
    """

    work = work or _WorkCounter()
    if model.active_expert_count >= model.max_experts:
        return ChallengerDecision(
            accepted=False,
            reason="capacity_exhausted",
            evidence=None,
            expert_parameters=None,
            router_weight=None,
            router_bias=None,
        )
    device = model.feature_mean.device
    features = proposal.features.to(device=device, dtype=torch.float32)
    labels = proposal.labels.to(device=device, dtype=torch.float32)
    fit_indices = proposal.fit_indices.to(device=device)
    validation_indices = proposal.validation_indices.to(device=device)
    fit_features = features[fit_indices]
    fit_labels = labels[fit_indices]
    validation_features = features[validation_indices]
    validation_labels = labels[validation_indices]
    anchors = replay_features.detach().to(device=device, dtype=torch.float32)
    anchor_labels = replay_labels.detach().to(device=device, dtype=torch.float32)
    anchor_routes = replay_routes.detach().to(device=device, dtype=torch.long)
    anchor_fit_indices, anchor_validation_indices = _split_replay_anchors(
        anchor_routes,
        active_count=model.active_expert_count,
        generator=_generator(
            device,
            config.seed + 524_287 * (proposal.proposal_id + 1),
        ),
    )
    fit_anchors = anchors[anchor_fit_indices]
    fit_anchor_labels = anchor_labels[anchor_fit_indices]
    fit_anchor_routes = anchor_routes[anchor_fit_indices]
    validation_anchors = anchors[anchor_validation_indices]
    validation_anchor_labels = anchor_labels[anchor_validation_indices]
    validation_anchor_routes = anchor_routes[anchor_validation_indices]

    with torch.no_grad():
        fit_incumbent_logits = model.active_expert_logits(fit_features)
        fit_incumbent_losses = F.binary_cross_entropy_with_logits(
            fit_incumbent_logits,
            fit_labels.unsqueeze(0).expand_as(fit_incumbent_logits),
            reduction="none",
        )
        parent_expert = int(fit_incumbent_losses.mean(dim=1).argmin().item())
        parent_fit_logits = fit_incumbent_logits[parent_expert]
        incumbent_errors = (
            (parent_fit_logits >= 0) != fit_labels.to(dtype=torch.bool)
        ).sum()
        estimated_noise_rate = min(
            0.5 - config.noise_epsilon,
            (config.beta_prior_a + float(incumbent_errors.item()))
            / (config.beta_prior_a + config.beta_prior_b + fit_labels.numel()),
        )
        parent_validation_logits = model.active_expert_logits(validation_features)[
            parent_expert
        ]
        parent_validation_observed = torch.where(
            validation_labels.bool(),
            parent_validation_logits.sigmoid(),
            (-parent_validation_logits).sigmoid(),
        )
        noise_validation_probability = (
            estimated_noise_rate
            + (1 - 2 * estimated_noise_rate) * parent_validation_observed
        )
        noise_validation_losses = -noise_validation_probability.clamp_min(
            torch.finfo(noise_validation_probability.dtype).tiny
        ).log()
        parent_fit_observed = torch.where(
            fit_labels.bool(),
            parent_fit_logits.sigmoid(),
            (-parent_fit_logits).sigmoid(),
        )
        noise_fit_probability = (
            estimated_noise_rate + (1 - 2 * estimated_noise_rate) * parent_fit_observed
        )
        noise_fit_losses = -noise_fit_probability.clamp_min(
            torch.finfo(noise_fit_probability.dtype).tiny
        ).log()
    work.candidate_scoring_forward_examples += (
        fit_features.shape[0] + validation_features.shape[0]
    )

    provisional_router = _provisional_router(
        model=model,
        proposal_features=fit_features,
        anchor_features=fit_anchors,
        anchor_routes=fit_anchor_routes,
        config=config,
        seed=config.seed + 2_000_033 * (proposal.proposal_id + 1),
        work=work,
    )
    with torch.no_grad():
        if fit_anchors.shape[0]:
            standardized_fit_anchors = (
                fit_anchors - model.feature_mean
            ) / model.feature_scale
            candidate_anchor_mask = (
                provisional_router(standardized_fit_anchors).argmax(dim=-1)
                == model.active_expert_count
            )
        else:
            candidate_anchor_mask = torch.zeros(
                0,
                device=device,
                dtype=torch.bool,
            )
    candidate_anchors = fit_anchors[candidate_anchor_mask]
    candidate_anchor_labels = fit_anchor_labels[candidate_anchor_mask]
    candidate_anchor_routes = fit_anchor_routes[candidate_anchor_mask]

    challenger = _ProvisionalExpert(
        model,
        model.parent_parameters(parent_expert),
    ).to(device)
    optimizer = torch.optim.AdamW(
        challenger.parameters(),
        lr=config.challenger_learning_rate,
        weight_decay=0,
    )
    anchor_weights = _balanced_anchor_weights(
        candidate_anchor_routes,
        model.active_expert_count,
    ).to(device)
    for _ in range(config.challenger_steps):
        optimizer.zero_grad(set_to_none=True)
        fit_loss = F.binary_cross_entropy_with_logits(
            challenger(fit_features),
            fit_labels,
        )
        if candidate_anchors.shape[0]:
            anchor_losses = F.binary_cross_entropy_with_logits(
                challenger(candidate_anchors),
                candidate_anchor_labels,
                reduction="none",
            )
            loss = (
                fit_loss
                + config.challenger_anchor_weight
                * (anchor_losses * anchor_weights).sum()
            )
        else:
            loss = fit_loss
        loss.backward()
        optimizer.step()
    fit_work = fit_features.shape[0] + candidate_anchors.shape[0]
    work.candidate_fit_forward_examples += config.challenger_steps * fit_work
    work.candidate_fit_backward_examples += config.challenger_steps * fit_work

    with torch.no_grad():
        challenger_fit_losses = F.binary_cross_entropy_with_logits(
            challenger(fit_features),
            fit_labels,
            reduction="none",
        )
        challenger_validation_losses = F.binary_cross_entropy_with_logits(
            challenger(validation_features),
            validation_labels,
            reduction="none",
        )
        validation_improvements = noise_validation_losses - challenger_validation_losses
        mean_improvement = float(validation_improvements.mean().item())
        fit_improvement = float(
            (noise_fit_losses - challenger_fit_losses).mean().item()
        )
        lower_bound = _bootstrap_lower_bound(
            validation_improvements,
            samples=config.bootstrap_samples,
            alpha=config.confidence_alpha,
            generator=_generator(
                device,
                config.seed + 1_000_003 * (proposal.proposal_id + 1),
            ),
        )
    work.candidate_scoring_forward_examples += (
        fit_features.shape[0] + validation_features.shape[0]
    )

    with torch.no_grad():
        standardized_validation = (
            validation_features - model.feature_mean
        ) / model.feature_scale
        validation_routes = provisional_router(standardized_validation).argmax(dim=-1)
        new_expert = model.active_expert_count
        proposal_route_accuracy = float(
            (validation_routes == new_expert).float().mean().item()
        )
        if validation_anchors.shape[0]:
            standardized_anchors = (
                validation_anchors - model.feature_mean
            ) / model.feature_scale
            routed_anchors = provisional_router(standardized_anchors).argmax(dim=-1)
            preserved_anchors = routed_anchors != new_expert
            accuracy_mask = (
                preserved_anchors
                if config.routing_strategy == "prototype"
                else torch.ones_like(preserved_anchors)
            )
            anchor_route_accuracy = (
                float(
                    (
                        routed_anchors[accuracy_mask]
                        == validation_anchor_routes[accuracy_mask]
                    )
                    .float()
                    .mean()
                    .item()
                )
                if torch.any(accuracy_mask)
                else 1.0
            )
            baseline_logits = model.selected_expert_logits(
                validation_anchors,
                validation_anchor_routes,
            )
            challenger_anchor_logits = challenger(validation_anchors)
            provisional_logits = model.selected_expert_logits(
                validation_anchors,
                routed_anchors.clamp_max(model.active_expert_count - 1),
            )
            provisional_logits = torch.where(
                routed_anchors == new_expert,
                challenger_anchor_logits,
                provisional_logits,
            )
            newly_routed = routed_anchors == new_expert
            if torch.any(newly_routed):
                incumbent_anchor_losses = F.binary_cross_entropy_with_logits(
                    baseline_logits[newly_routed],
                    validation_anchor_labels[newly_routed],
                    reduction="none",
                )
                incumbent_observed_probability = torch.where(
                    validation_anchor_labels[newly_routed].bool(),
                    baseline_logits[newly_routed].sigmoid(),
                    (-baseline_logits[newly_routed]).sigmoid(),
                )
                noise_anchor_probability = (
                    estimated_noise_rate
                    + (1 - 2 * estimated_noise_rate) * incumbent_observed_probability
                )
                noise_anchor_losses = -noise_anchor_probability.clamp_min(
                    torch.finfo(noise_anchor_probability.dtype).tiny
                ).log()
                provisional_anchor_losses = F.binary_cross_entropy_with_logits(
                    provisional_logits[newly_routed],
                    validation_anchor_labels[newly_routed],
                    reduction="none",
                )
                collateral_improvements = (
                    noise_anchor_losses - provisional_anchor_losses
                )
                collateral_mean_improvement = float(
                    collateral_improvements.mean().item()
                )
                collateral_change = float(
                    (provisional_anchor_losses - incumbent_anchor_losses).mean().item()
                )
                collateral_lower_bound = _bootstrap_lower_bound(
                    collateral_improvements,
                    samples=config.bootstrap_samples,
                    alpha=config.confidence_alpha,
                    generator=_generator(
                        device,
                        config.seed + 4_000_037 * (proposal.proposal_id + 1),
                    ),
                )
                collateral_support = int(newly_routed.sum().item())
            else:
                collateral_change = 0.0
                collateral_support = 0
                collateral_mean_improvement = 0.0
                collateral_lower_bound = 0.0
            work.candidate_scoring_forward_examples += 3 * validation_anchors.shape[0]
        else:
            anchor_route_accuracy = 1.0
            collateral_change = 0.0
            collateral_support = 0
            collateral_mean_improvement = 0.0
            collateral_lower_bound = 0.0
    work.candidate_scoring_forward_examples += validation_features.shape[0]
    has_both_classes = bool(torch.unique(validation_labels).numel() == 2)
    rule_log_bayes_factor = float(validation_improvements.sum().item())
    if collateral_support:
        rule_log_bayes_factor += float(collateral_improvements.sum().item())
    prior_log_odds = math.log(config.context_hazard) - math.log1p(
        -config.context_hazard
    )
    posterior_log_odds = prior_log_odds + rule_log_bayes_factor
    if posterior_log_odds >= 0:
        unexpected_uncertainty = 1 / (1 + math.exp(-posterior_log_odds))
    else:
        posterior_odds = math.exp(posterior_log_odds)
        unexpected_uncertainty = posterior_odds / (1 + posterior_odds)
    expected_odds = estimated_noise_rate / (1 - estimated_noise_rate)
    switch_factor = model.active_expert_count / (
        config.context_persistence * (1 - config.context_persistence)
    )
    context_switch_log_margin = posterior_log_odds - math.log(
        expected_odds * switch_factor
    )
    evidence = BirthEvidence(
        parent_expert=parent_expert,
        fit_support=fit_features.shape[0],
        validation_support=validation_features.shape[0],
        estimated_noise_rate=estimated_noise_rate,
        mean_improvement=mean_improvement,
        lower_confidence_bound=lower_bound,
        fit_improvement=fit_improvement,
        validation_has_both_classes=has_both_classes,
        proposal_route_accuracy=proposal_route_accuracy,
        anchor_route_accuracy=anchor_route_accuracy,
        collateral_loss_change=collateral_change,
        collateral_support=collateral_support,
        collateral_mean_improvement=collateral_mean_improvement,
        collateral_lower_confidence_bound=collateral_lower_bound,
        rule_log_bayes_factor=rule_log_bayes_factor,
        unexpected_uncertainty=unexpected_uncertainty,
        context_switch_log_margin=context_switch_log_margin,
    )

    if mode == "raw_loss":
        accepted = True
        reason = "accepted_raw_coherence"
    elif mode == "unvalidated":
        accepted = fit_improvement > config.practical_margin
        reason = (
            "accepted_in_sample" if accepted else "insufficient_in_sample_improvement"
        )
    elif proposal_route_accuracy < config.router_min_proposal_accuracy:
        accepted = False
        reason = "route_overlap"
    elif anchor_route_accuracy < config.router_min_anchor_accuracy:
        accepted = False
        reason = "anchor_route_regression"
    elif collateral_change > config.collateral_tolerance:
        accepted = False
        reason = "collateral_regression"
    elif not has_both_classes:
        accepted = False
        reason = "validation_single_class"
    elif collateral_support < config.collateral_min_support:
        accepted = False
        reason = "insufficient_collateral_support"
    elif collateral_mean_improvement <= config.collateral_practical_margin:
        accepted = False
        reason = "collateral_not_learnable"
    elif mean_improvement <= config.practical_margin:
        accepted = False
        reason = "insufficient_practical_improvement"
    elif unexpected_uncertainty < config.birth_posterior_threshold:
        accepted = False
        reason = "insufficient_rule_posterior"
    elif context_switch_log_margin <= 0:
        accepted = False
        reason = "expected_uncertainty_dominates"
    else:
        accepted = True
        reason = "accepted_held_out_rule"

    if accepted:
        # Once the decision is fixed, all accepted pseudo-labels may supervise
        # the final copied router. Outcome labels in the validation partition
        # still never entered challenger optimization.
        provisional_router = _provisional_router(
            model=model,
            proposal_features=features,
            anchor_features=fit_anchors,
            anchor_routes=fit_anchor_routes,
            config=config,
            seed=config.seed + 3_000_017 * (proposal.proposal_id + 1),
            work=work,
        )
    return ChallengerDecision(
        accepted=accepted,
        reason=reason,
        evidence=evidence,
        expert_parameters=challenger.detached_parameters(),
        router_weight=provisional_router.weight.detach().clone(),
        router_bias=provisional_router.bias.detach().clone(),
    )


class _ReplayReservoir:
    def __init__(self, capacity: int, generator: torch.Generator) -> None:
        self.capacity = capacity
        self.generator = generator
        self.seen = 0
        self.features: list[Tensor] = []
        self.labels: list[Tensor] = []
        self.routes: list[Tensor] = []

    def add(self, features: Tensor, labels: Tensor, routes: Tensor) -> None:
        for index in range(labels.numel()):
            self.seen += 1
            if len(self.features) < self.capacity:
                position = len(self.features)
                self.features.append(features[index].detach().cpu().clone())
                self.labels.append(labels[index].detach().cpu().clone())
                self.routes.append(routes[index].detach().cpu().clone())
            else:
                replacement = int(
                    torch.randint(
                        self.seen,
                        (),
                        generator=self.generator,
                    ).item()
                )
                if replacement >= self.capacity:
                    continue
                position = replacement
                self.features[position] = features[index].detach().cpu().clone()
                self.labels[position] = labels[index].detach().cpu().clone()
                self.routes[position] = routes[index].detach().cpu().clone()

    def tensors(self, width: int) -> tuple[Tensor, Tensor, Tensor]:
        if not self.features:
            return (
                torch.empty((0, width), dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
            )
        return (
            torch.stack(self.features),
            torch.stack(self.labels),
            torch.stack(self.routes).long(),
        )


def _validate_config(config: SpawningMoEConfig) -> None:
    if config.expert_architecture not in {"stable_latent", "plain"}:
        raise ValueError(f"unknown expert architecture: {config.expert_architecture!r}")
    if config.routing_strategy not in {"learned", "prototype"}:
        raise ValueError(f"unknown routing strategy: {config.routing_strategy!r}")
    positive_integers = (
        config.max_experts,
        config.hidden_dimensions,
        config.latent_dimensions,
        config.batch_size,
        config.warmup_count,
        config.calibration_capacity,
        config.proposal_interval,
        config.proposal_clusters,
        config.proposal_min_support,
        config.proposal_buffer_capacity,
        config.kmeans_iterations,
        config.challenger_steps,
        config.bootstrap_samples,
        config.router_steps,
        config.replay_capacity,
        config.collateral_min_support,
    )
    if min(positive_integers) < 1:
        raise ValueError("model, buffer, iteration, and support sizes must be positive")
    if config.proposal_min_support > config.proposal_buffer_capacity:
        raise ValueError("proposal support cannot exceed buffer capacity")
    if not 0 < config.surprise_tail_probability < 1:
        raise ValueError("surprise_tail_probability must lie in (0, 1)")
    if not 0 < config.proposal_validation_fraction < 1:
        raise ValueError("proposal_validation_fraction must lie in (0, 1)")
    if not 0 < config.confidence_alpha < 0.5:
        raise ValueError("confidence_alpha must lie in (0, 0.5)")
    if not 0 < config.noise_epsilon < 0.5:
        raise ValueError("noise_epsilon must lie in (0, 0.5)")
    if not 0 < config.context_hazard < 1:
        raise ValueError("context_hazard must lie in (0, 1)")
    if not 0 < config.context_persistence < 1:
        raise ValueError("context_persistence must lie in (0, 1)")
    if not 0 < config.birth_posterior_threshold < 1:
        raise ValueError("birth_posterior_threshold must lie in (0, 1)")
    if (
        min(
            config.beta_prior_a,
            config.beta_prior_b,
            config.learning_rate,
            config.challenger_learning_rate,
            config.router_learning_rate,
            config.prototype_variance_floor,
        )
        <= 0
    ):
        raise ValueError("priors and learning rates must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    bounded = (
        config.router_min_proposal_accuracy,
        config.router_min_anchor_accuracy,
    )
    if any(not 0 <= value <= 1 for value in bounded):
        raise ValueError("router accuracy thresholds must lie in [0, 1]")


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


def _validate_training_tensors(
    train_features: Tensor,
    observed_labels: Tensor,
    test_features: Tensor,
) -> None:
    if not all(
        isinstance(tensor, Tensor)
        for tensor in (train_features, observed_labels, test_features)
    ):
        raise TypeError("features and observed_labels must be tensors")
    if train_features.ndim != 2 or train_features.shape[0] == 0:
        raise ValueError("train_features must be a non-empty matrix")
    if test_features.ndim != 2:
        raise ValueError("test_features must be a matrix")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("train and test feature widths must match")
    if observed_labels.shape != (train_features.shape[0],):
        raise ValueError("observed_labels must contain one scalar per train example")
    if not all(
        torch.all(torch.isfinite(tensor))
        for tensor in (train_features, observed_labels, test_features)
    ):
        raise ValueError("training tensors must be finite")
    if torch.any((observed_labels != 0) & (observed_labels != 1)):
        raise ValueError("observed_labels must be binary")


def train_spawning_moe(
    train_features: Tensor,
    observed_labels: Tensor,
    test_features: Tensor,
    *,
    mode: SpawnerMode = "joint",
    config: SpawningMoEConfig | None = None,
    device: str | torch.device | None = None,
) -> SpawningTrainingResult:
    """Train one deterministic online spawning arm from observed tensors only."""

    if mode not in {
        "single",
        "raw_loss",
        "expected_only",
        "unvalidated",
        "joint",
    }:
        raise ValueError(f"unknown spawning mode: {mode!r}")
    config = config or SpawningMoEConfig()
    _validate_config(config)
    _validate_training_tensors(train_features, observed_labels, test_features)
    selected_device = _resolve_device(
        train_features,
        config.device if device is None else device,
    )
    training = train_features.detach().to(selected_device, dtype=torch.float32)
    labels = observed_labels.detach().to(selected_device, dtype=torch.float32)
    testing = test_features.detach().to(selected_device, dtype=torch.float32)
    feature_mean = training.mean(dim=0)
    feature_scale = (
        (training - feature_mean).square().mean(dim=0).sqrt().clamp_min(1e-6)
    )
    model = SpawningMoE(
        input_dimensions=training.shape[1],
        hidden_dimensions=config.hidden_dimensions,
        latent_dimensions=config.latent_dimensions,
        expert_architecture=config.expert_architecture,
        max_experts=config.max_experts,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    ).to(selected_device)
    model.initialize(_generator(selected_device, config.seed))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    controller = UnexpectedUncertaintyController(config, mode)
    work = _WorkCounter()
    replay = _ReplayReservoir(
        config.replay_capacity,
        torch.Generator().manual_seed(config.seed + 7_919),
    )
    # The caller owns the deterministic stream order. This preserves exact
    # prefixes in nested-support experiments instead of reshuffling every
    # budget with a length-dependent permutation.
    stream = torch.arange(
        training.shape[0],
        device=selected_device,
    )

    for start in range(0, stream.numel(), config.batch_size):
        indices = stream[start : start + config.batch_size]
        batch_features = training[indices]
        batch_labels = labels[indices]
        with torch.no_grad():
            preupdate_logits = model.active_expert_logits(batch_features)
            preupdate_losses = F.binary_cross_entropy_with_logits(
                preupdate_logits,
                batch_labels.unsqueeze(0).expand_as(preupdate_logits),
                reduction="none",
            )
        work.controller_scoring_forward_examples += (
            model.active_expert_count * batch_labels.numel()
        )
        proposals = controller.observe_batch(
            example_ids=indices,
            features=batch_features,
            labels=batch_labels,
            active_losses=preupdate_losses,
            work=work,
        )
        for proposal in proposals:
            replay_features, replay_labels, replay_routes = replay.tensors(
                training.shape[1]
            )
            decision = evaluate_birth_proposal(
                model,
                proposal,
                replay_features,
                replay_labels,
                replay_routes,
                mode=mode,  # type: ignore[arg-type]
                config=config,
                work=work,
            )
            if (
                decision.accepted
                and decision.expert_parameters is not None
                and decision.router_weight is not None
                and decision.router_bias is not None
            ):
                expert_id = model.activate(
                    expert_parameters=decision.expert_parameters,
                    router_weight=decision.router_weight,
                    router_bias=decision.router_bias,
                )
                controller.resolve(proposal, decision, expert_id=expert_id)
            else:
                controller.resolve(proposal, decision)

        optimizer.zero_grad(set_to_none=True)
        routed_logits, routes = model(batch_features)
        task_loss = F.binary_cross_entropy_with_logits(
            routed_logits,
            batch_labels,
        )
        task_loss.backward()
        optimizer.step()
        work.task_forward_examples += batch_labels.numel()
        work.task_backward_examples += batch_labels.numel()
        replay.add(batch_features, batch_labels, routes)

    model.eval()
    with torch.no_grad():
        routed_logits, routed_experts = model(testing)
        routed_probabilities = routed_logits.sigmoid()
        route_counts = torch.bincount(
            routed_experts,
            minlength=config.max_experts,
        )
    work.sparse_inference_examples += testing.shape[0]
    compute = work.freeze()
    return SpawningTrainingResult(
        mode=mode,
        config=config,
        device=selected_device,
        predictions=SpawningPredictions(
            routed_logits=routed_logits,
            routed_probabilities=routed_probabilities,
            routed_expert_indices=routed_experts,
            route_counts=route_counts,
            active_expert_mask=model.active_expert_mask.detach().clone(),
        ),
        births=tuple(controller.births),
        rejections=tuple(controller.rejections),
        calibration_counts=controller.calibration.counts(),
        expected_uncertainties=tuple(
            controller.calibration.expected_uncertainty(expert)
            for expert in range(config.max_experts)
        ),
        surprising_examples=controller.surprise_count,
        proposal_count=controller.next_proposal_id,
        unresolved_count=controller.unresolved_count,
        compute=compute,
        diagnostics_json=controller.diagnostics_json(),
    )
