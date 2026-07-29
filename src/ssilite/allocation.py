"""Sampling and precision allocation for a robust gradient estimator."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import pairwise

import torch
from torch import Tensor


def _simplex_vector(name: str, values: Tensor) -> None:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not values.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if not torch.all(torch.isfinite(values)) or torch.any(values < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    total = values.to(dtype=torch.float64).sum()
    if not torch.isclose(total, total.new_tensor(1.0), atol=1e-8):
        raise ValueError(f"{name} must sum to one")


@torch.no_grad()
def variance_aware_sampling_probabilities(
    robust_weights: Tensor,
    gradient_norm_sq: Tensor,
    quantization_variance: Tensor | None = None,
    *,
    defensive_mass: float = 0.1,
    exploration: float = 0.05,
) -> Tensor:
    """Construct the variance-minimizing proposal with defensive mixtures.

    Before stabilization, ``p_i`` is proportional to
    ``q_i * sqrt(||g_i||^2 + sigma_i^2)``.  Mixing with ``q`` bounds importance
    ratios, while uniform exploration preserves support.
    """

    _simplex_vector("robust_weights", robust_weights)
    if gradient_norm_sq.shape != robust_weights.shape:
        raise ValueError("gradient_norm_sq must match robust_weights")
    if not torch.all(torch.isfinite(gradient_norm_sq)) or torch.any(
        gradient_norm_sq < 0
    ):
        raise ValueError("gradient_norm_sq must be finite and non-negative")
    if quantization_variance is None:
        quantization_variance = torch.zeros_like(gradient_norm_sq)
    if quantization_variance.shape != robust_weights.shape:
        raise ValueError("quantization_variance must match robust_weights")
    if not torch.all(torch.isfinite(quantization_variance)) or torch.any(
        quantization_variance < 0
    ):
        raise ValueError("quantization_variance must be finite and non-negative")
    if not math.isfinite(defensive_mass) or not math.isfinite(exploration):
        raise ValueError("mixture masses must be finite")
    if defensive_mass < 0 or exploration < 0:
        raise ValueError("mixture masses must be non-negative")
    if defensive_mass + exploration >= 1:
        raise ValueError("mixture masses must sum to less than one")

    working_weights = robust_weights.to(dtype=torch.float64)
    working_gradient = gradient_norm_sq.to(dtype=torch.float64)
    working_variance = quantization_variance.to(dtype=torch.float64)
    normalizer = torch.maximum(
        working_gradient.max(), working_variance.max()
    ).clamp_min(1)
    scale = (working_gradient / normalizer + working_variance / normalizer).sqrt()
    raw = working_weights * scale
    if raw.sum() <= 0:
        raw = robust_weights.clone()
    raw /= raw.sum()
    uniform = torch.full_like(raw, 1 / raw.numel())
    probabilities = (
        (1 - defensive_mass - exploration) * raw
        + defensive_mass * robust_weights
        + exploration * uniform
    )
    return (probabilities / probabilities.sum()).to(dtype=robust_weights.dtype).detach()


@dataclass(frozen=True)
class PrecisionAllocation:
    """Discrete precision choices and their resulting expected cost."""

    bits: Tensor
    level_indices: Tensor
    expected_cost: float
    variance_term: float


@torch.no_grad()
def greedy_precision_allocation(
    robust_weights: Tensor,
    sampling_probabilities: Tensor,
    quantization_variances: Tensor,
    costs: Tensor,
    levels: Sequence[int],
    *,
    mean_cost_budget: float,
) -> PrecisionAllocation:
    """Greedily buy precision upgrades by variance reduction per unit cost.

    ``quantization_variances[i, k]`` models the quantization contribution for
    example ``i`` at precision level ``k``.  The optimized term is
    ``sum_i q_i**2 / p_i * variance[i, selected_i]`` under the expected
    ``p``-weighted cost budget.
    """

    _simplex_vector("robust_weights", robust_weights)
    _simplex_vector("sampling_probabilities", sampling_probabilities)
    count = robust_weights.numel()
    if sampling_probabilities.shape != robust_weights.shape:
        raise ValueError("sampling_probabilities must match robust_weights")
    if torch.any(sampling_probabilities <= 0):
        raise ValueError("sampling_probabilities must be strictly positive")
    if not math.isfinite(mean_cost_budget):
        raise ValueError("mean_cost_budget must be finite")
    level_count = len(levels)
    if level_count == 0 or any(level < 2 for level in levels):
        raise ValueError("levels must contain bit widths of at least two")
    if any(right <= left for left, right in pairwise(levels)):
        raise ValueError("levels must be strictly increasing")
    if quantization_variances.shape != (count, level_count):
        raise ValueError("quantization_variances must have shape [n, levels]")
    if not torch.all(torch.isfinite(quantization_variances)) or torch.any(
        quantization_variances < 0
    ):
        raise ValueError("quantization variances must be finite and non-negative")
    if torch.any(quantization_variances[:, 1:] > quantization_variances[:, :-1]):
        raise ValueError("quantization variance must not increase with precision")

    working_weights = robust_weights.detach().to(dtype=torch.float64)
    working_probabilities = sampling_probabilities.detach().to(dtype=torch.float64)
    working_variances = quantization_variances.detach().to(
        device=robust_weights.device, dtype=torch.float64
    )
    working_costs = costs.detach().to(device=robust_weights.device, dtype=torch.float64)

    if working_costs.ndim == 1:
        if working_costs.numel() != level_count:
            raise ValueError("one-dimensional costs must have one value per level")
        expanded_costs = working_costs.unsqueeze(0).expand(count, -1)
    elif working_costs.shape == (count, level_count):
        expanded_costs = working_costs
    else:
        raise ValueError("costs must have shape [levels] or [n, levels]")
    if not torch.all(torch.isfinite(expanded_costs)) or torch.any(expanded_costs < 0):
        raise ValueError("costs must be finite and non-negative")
    if torch.any(expanded_costs[:, 1:] < expanded_costs[:, :-1]):
        raise ValueError("costs must not decrease with precision")

    selected = torch.zeros(count, dtype=torch.long, device=robust_weights.device)

    def expected_cost(indices: Tensor) -> Tensor:
        row = torch.arange(count, device=indices.device)
        return torch.dot(working_probabilities, expanded_costs[row, indices])

    minimum_cost = expected_cost(selected)
    maximum_indices = torch.full_like(selected, level_count - 1)
    maximum_cost = expected_cost(maximum_indices)
    tolerance = 1e-12 * max(1.0, abs(mean_cost_budget))
    if mean_cost_budget + tolerance < float(minimum_cost.item()):
        raise ValueError("mean_cost_budget is below the minimum feasible cost")
    budget = min(mean_cost_budget, float(maximum_cost.item()))

    priority_queue: list[tuple[float, int, int, float]] = []

    def push_upgrade(index: int, level: int) -> None:
        if level + 1 >= level_count:
            return
        delta_cost = working_probabilities[index] * (
            expanded_costs[index, level + 1] - expanded_costs[index, level]
        )
        delta_variance = (
            working_weights[index].square()
            / working_probabilities[index]
            * (working_variances[index, level] - working_variances[index, level + 1])
        )
        ratio = (
            float("inf")
            if delta_cost <= 0
            else float((delta_variance / delta_cost).item())
        )
        heappush(
            priority_queue,
            (-ratio, index, level, float(delta_cost.item())),
        )

    for index in range(count):
        push_upgrade(index, 0)

    current_cost = float(minimum_cost.item())
    while priority_queue:
        _, index, level, delta_cost = heappop(priority_queue)
        if int(selected[index].item()) != level:
            continue
        if current_cost + delta_cost > budget:
            continue
        selected[index] += 1
        current_cost += delta_cost
        push_upgrade(index, level + 1)

    rows = torch.arange(count, device=selected.device)
    selected_variances = working_variances[rows, selected]
    variance_term = (
        working_weights.square() / working_probabilities * selected_variances
    ).sum()
    level_tensor = torch.tensor(levels, dtype=torch.long, device=selected.device)
    return PrecisionAllocation(
        bits=level_tensor[selected],
        level_indices=selected,
        expected_cost=float(expected_cost(selected).item()),
        variance_term=float(variance_term.item()),
    )
