"""Joint within-support controller for robust weights, sampling, and precision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .allocation import (
    greedy_precision_allocation,
    variance_aware_sampling_probabilities,
)
from .quantization import deterministic_quantize, quantization_variance_proxy
from .risk import capped_entropic_weights


@dataclass(frozen=True)
class ControllerConfig:
    tail_fraction: float = 0.25
    temperature: float = 0.1
    dual_step: float = 0.05
    score_bits: int = 4
    defensive_mass: float = 0.1
    exploration: float = 0.05
    precision_levels: tuple[int, ...] = (4, 8, 16)
    precision_costs: tuple[float, ...] = (1.0, 2.0, 4.0)
    mean_precision_budget: float = 2.0
    allocation_rounds: int = 3

    def __post_init__(self) -> None:
        if not 0 < self.dual_step <= 1:
            raise ValueError("dual_step must lie in (0, 1]")
        if len(self.precision_levels) != len(self.precision_costs):
            raise ValueError("precision_levels and precision_costs must match")
        if self.allocation_rounds < 1:
            raise ValueError("allocation_rounds must be positive")


@dataclass(frozen=True)
class BatchAllocation:
    indices: Tensor
    robust_weights: Tensor
    sampling_probabilities: Tensor
    precision_bits: Tensor
    importance_weights: Tensor
    expected_precision_cost: float
    robust_score: float | None
    max_importance_weight: float | None


class JointController:
    """Stateful, damped controller over a fixed finite support."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self._robust_weights: Tensor | None = None
        self._base_weights: Tensor | None = None

    def reset(self) -> None:
        self._robust_weights = None
        self._base_weights = None

    @torch.no_grad()
    def update_robust_weights(
        self,
        scores: Tensor,
        *,
        base_weights: Tensor | None = None,
    ) -> Tensor:
        """Update and return robust weights without sampling or precision work."""

        if scores.ndim != 1 or scores.numel() == 0:
            raise ValueError("scores must be a non-empty vector")

        quantized_scores = deterministic_quantize(
            scores.detach(), self.config.score_bits
        )
        target_weights = capped_entropic_weights(
            quantized_scores,
            self.config.tail_fraction,
            self.config.temperature,
            base_weights=base_weights,
        )
        if base_weights is None:
            base = torch.full_like(target_weights, 1 / target_weights.numel())
        else:
            base = base_weights.detach().to(dtype=target_weights.dtype)
            base = base / base.sum()
        if (
            self._robust_weights is None
            or self._robust_weights.shape != target_weights.shape
            or self._robust_weights.device != target_weights.device
            or self._robust_weights.dtype != target_weights.dtype
            or self._base_weights is None
            or not torch.equal(self._base_weights, base)
        ):
            self._robust_weights = base
        robust_weights = (
            1 - self.config.dual_step
        ) * self._robust_weights + self.config.dual_step * target_weights
        robust_weights /= robust_weights.sum()
        self._robust_weights = robust_weights.detach()
        self._base_weights = base.detach()
        return robust_weights

    @torch.no_grad()
    def allocate(
        self,
        scores: Tensor,
        gradient_norm_sq_proxy: Tensor,
        batch_size: int,
        *,
        base_weights: Tensor | None = None,
        generator: torch.Generator | None = None,
        diagnostics: bool = True,
    ) -> BatchAllocation:
        """Allocate a with-replacement training batch and its precision.

        Controller statistics are derived before the random batch draw.  Given
        this detached state, ``q[indices] / p[indices]`` is the correct
        importance ratio for an unbiased estimator of the robust gradient.
        Setting ``diagnostics=False`` leaves the scalar diagnostic fields unset
        and avoids their device-to-host synchronizations.
        """

        if gradient_norm_sq_proxy.shape != scores.shape:
            raise ValueError("gradient_norm_sq_proxy must match scores")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        robust_weights = self.update_robust_weights(
            scores,
            base_weights=base_weights,
        )

        levels = self.config.precision_levels
        variance_table = torch.stack(
            [
                quantization_variance_proxy(gradient_norm_sq_proxy, level)
                for level in levels
            ],
            dim=1,
        )
        current_variance = variance_table[:, 0]
        probabilities = robust_weights
        precision = None
        costs = scores.new_tensor(self.config.precision_costs)
        if scores.is_cuda:
            allocation_weights = robust_weights.cpu()
            allocation_variances = variance_table.cpu()
            allocation_costs = costs.cpu()
        else:
            allocation_weights = robust_weights
            allocation_variances = variance_table
            allocation_costs = costs
        for _ in range(self.config.allocation_rounds):
            probabilities = variance_aware_sampling_probabilities(
                robust_weights,
                gradient_norm_sq_proxy,
                current_variance,
                defensive_mass=self.config.defensive_mass,
                exploration=self.config.exploration,
            )
            precision = greedy_precision_allocation(
                allocation_weights,
                probabilities.cpu() if scores.is_cuda else probabilities,
                allocation_variances,
                allocation_costs,
                levels,
                mean_cost_budget=self.config.mean_precision_budget,
            )
            level_indices = precision.level_indices.to(device=scores.device)
            rows = torch.arange(scores.numel(), device=scores.device)
            current_variance = variance_table[rows, level_indices]

        assert precision is not None
        precision_bits = precision.bits.to(device=scores.device)
        level_indices = precision.level_indices.to(device=scores.device)
        indices = torch.multinomial(
            probabilities,
            batch_size,
            replacement=True,
            generator=generator,
        )
        importance_weights = robust_weights[indices] / probabilities[indices]
        actual_cost = (
            precision.expected_cost
            if scores.is_cuda
            else float(torch.dot(probabilities, costs[level_indices]).item())
        )
        robust_score = (
            float(torch.dot(robust_weights, scores).item()) if diagnostics else None
        )
        max_importance_weight = (
            float(importance_weights.max().item()) if diagnostics else None
        )
        return BatchAllocation(
            indices=indices,
            robust_weights=robust_weights,
            sampling_probabilities=probabilities,
            precision_bits=precision_bits,
            importance_weights=importance_weights,
            expected_precision_cost=float(actual_cost),
            robust_score=robust_score,
            max_importance_weight=max_importance_weight,
        )
