"""A bounded adversary over discovered environments.

Per-example loss is a ranking and cannot recover a partition.  This controller
instead receives a label-free environment assignment and maintains a dual
distribution over environment risks.  Within each environment, a separately
estimated trust score filters examples that no held-out student can reproduce.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .risk import capped_entropic_weights


@dataclass(frozen=True)
class EnvironmentAdversaryConfig:
    """Geometry and damping for the environment-level dual."""

    tail_fraction: float = 0.5
    temperature: float = 0.1
    dual_step: float = 0.1

    def __post_init__(self) -> None:
        if not 0 < self.tail_fraction <= 1:
            raise ValueError("tail_fraction must lie in (0, 1]")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.dual_step <= 1:
            raise ValueError("dual_step must lie in (0, 1]")


@dataclass(frozen=True)
class EnvironmentAllocation:
    """Environment risks, dual mass, and induced per-example objective."""

    environment_risks: Tensor
    environment_weights: Tensor
    example_weights: Tensor


def _validate_partition(assignments: Tensor, trust: Tensor) -> int:
    if assignments.ndim != 1 or assignments.numel() == 0:
        raise ValueError("assignments must be a non-empty vector")
    if assignments.dtype != torch.long:
        raise TypeError("assignments must use dtype torch.long")
    if trust.shape != assignments.shape:
        raise ValueError("trust must match assignments")
    if not trust.is_floating_point():
        raise TypeError("trust must use a floating-point dtype")
    if assignments.device != trust.device:
        raise ValueError("assignments and trust must share a device")
    if torch.any(assignments < 0):
        raise ValueError("assignments must be non-negative")
    if not torch.all(torch.isfinite(trust)) or torch.any(trust < 0):
        raise ValueError("trust must be finite and non-negative")
    environment_count = int(assignments.max().item()) + 1
    counts = torch.bincount(assignments, minlength=environment_count)
    if torch.any(counts == 0):
        raise ValueError("environment identifiers must be contiguous and non-empty")
    trust_mass = torch.zeros(
        environment_count,
        device=trust.device,
        dtype=trust.dtype,
    ).scatter_add_(0, assignments, trust)
    if torch.any(trust_mass <= 0):
        raise ValueError("every environment must have positive trust mass")
    return environment_count


@torch.no_grad()
def equal_environment_weights(assignments: Tensor, trust: Tensor) -> Tensor:
    """Give every discovered environment equal mass, modulated by trust."""

    environment_count = _validate_partition(assignments, trust)
    trust_mass = torch.zeros(
        environment_count,
        device=trust.device,
        dtype=trust.dtype,
    ).scatter_add_(0, assignments, trust)
    weights = trust / trust_mass[assignments] / environment_count
    return (weights / weights.sum()).detach()


class EnvironmentAdversary:
    """Damped capped-entropic maximization over discovered environment risks."""

    def __init__(
        self,
        assignments: Tensor,
        trust: Tensor,
        config: EnvironmentAdversaryConfig | None = None,
    ) -> None:
        self.config = config or EnvironmentAdversaryConfig()
        environment_count = _validate_partition(assignments, trust)
        self.assignments = assignments.detach()
        self.trust = trust.detach()
        self.environment_count = environment_count
        self._trust_mass = torch.zeros(
            environment_count,
            device=trust.device,
            dtype=trust.dtype,
        ).scatter_add_(0, assignments, trust)
        self._environment_weights = torch.full(
            (environment_count,),
            1 / environment_count,
            device=trust.device,
            dtype=trust.dtype,
        )

    @torch.no_grad()
    def update(self, losses: Tensor) -> EnvironmentAllocation:
        """Update the dual and return the induced example objective."""

        if losses.shape != self.assignments.shape:
            raise ValueError("losses must match the environment assignments")
        if losses.device != self.assignments.device:
            raise ValueError("losses and assignments must share a device")
        if not losses.is_floating_point():
            raise TypeError("losses must use a floating-point dtype")
        if not torch.all(torch.isfinite(losses)):
            raise ValueError("losses must be finite")

        weighted_loss = self.trust * losses
        risk_sums = torch.zeros(
            self.environment_count,
            device=losses.device,
            dtype=losses.dtype,
        ).scatter_add_(0, self.assignments, weighted_loss)
        environment_risks = risk_sums / self._trust_mass
        target = capped_entropic_weights(
            environment_risks,
            self.config.tail_fraction,
            self.config.temperature,
        )
        environment_weights = (
            1 - self.config.dual_step
        ) * self._environment_weights + self.config.dual_step * target
        environment_weights /= environment_weights.sum()
        self._environment_weights = environment_weights.detach()

        example_weights = (
            environment_weights[self.assignments]
            * self.trust
            / self._trust_mass[self.assignments]
        )
        example_weights /= example_weights.sum()
        return EnvironmentAllocation(
            environment_risks=environment_risks.detach(),
            environment_weights=environment_weights.detach(),
            example_weights=example_weights.detach(),
        )
