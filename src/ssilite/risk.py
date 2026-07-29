"""Bounded empirical tail-risk objectives."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _validate_scores(scores: Tensor, alpha: float) -> None:
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty vector")
    if not scores.is_floating_point():
        raise TypeError("scores must use a floating-point dtype")
    if not torch.all(torch.isfinite(scores)):
        raise ValueError("scores must be finite")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must lie in (0, 1]")


def _weight_cap(count: int, alpha: float) -> float:
    return min(1.0, 1.0 / (alpha * count))


@torch.no_grad()
def empirical_cvar_weights(scores: Tensor, alpha: float) -> Tensor:
    """Solve the empirical CVaR inner linear program exactly.

    The returned weights maximize ``dot(q, scores)`` subject to ``q`` lying on
    the simplex and ``q_i <= 1 / (alpha * n)``.  Mass at a tied threshold is
    split equally, making the solution permutation equivariant.
    """

    _validate_scores(scores, alpha)
    cap = _weight_cap(scores.numel(), alpha)
    working = scores.to(dtype=torch.float64)
    weights = torch.zeros_like(working)
    remaining = 1.0

    for level in torch.unique(working, sorted=True).flip(0):
        members = working == level
        member_count = int(members.sum().item())
        mass = min(remaining, cap * member_count)
        weights[members] = mass / member_count
        remaining -= mass
        if remaining <= 0:
            break

    return weights.to(dtype=scores.dtype).detach()


@torch.no_grad()
def capped_entropic_weights(
    scores: Tensor,
    alpha: float,
    temperature: float,
    *,
    base_weights: Tensor | None = None,
    bisection_steps: int = 80,
) -> Tensor:
    """Return a smooth, capped exponential tilt of empirical scores.

    This solves the entropy-regularized CVaR inner problem.  Its solution has
    ``q_i = min(base_i / alpha, c * base_i * exp(score_i / temperature))``;
    the scalar ``c`` is found by bisection.  When ``base_weights`` is omitted,
    the empirical uniform base is used.  The computation is performed in
    float64 for stability and the detached result is cast back to the input
    dtype.
    """

    _validate_scores(scores, alpha)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if bisection_steps < 1:
        raise ValueError("bisection_steps must be positive")

    working = scores.to(dtype=torch.float64)
    if base_weights is None:
        base = torch.full_like(working, 1 / working.numel())
    else:
        if base_weights.shape != scores.shape:
            raise ValueError("base_weights must match scores")
        if not base_weights.is_floating_point():
            raise TypeError("base_weights must use a floating-point dtype")
        if base_weights.device != scores.device:
            raise ValueError("base_weights must be on the same device as scores")
        if not torch.all(torch.isfinite(base_weights)) or torch.any(base_weights < 0):
            raise ValueError("base_weights must be finite and non-negative")
        base = base_weights.detach().to(dtype=torch.float64)
        total = base.sum()
        if not torch.isclose(total, total.new_tensor(1.0), atol=1e-8):
            raise ValueError("base_weights must sum to one")
        base = base / total

    support = base > 0
    logits = (working - working[support].max()) / temperature
    supported_logits = logits[support]
    supported_base = base[support]
    log_ratio_cap = working.new_tensor(-math.log(alpha))
    log_partition = torch.logsumexp(
        supported_base.log() + supported_logits,
        dim=0,
    )

    lower = -log_partition - 2
    upper = log_ratio_cap - supported_logits.min() + 2
    for _ in range(bisection_steps):
        midpoint = (lower + upper) / 2
        log_ratios = torch.minimum(log_ratio_cap, midpoint + supported_logits)
        total = torch.dot(supported_base, log_ratios.exp())
        move_lower = total < 1
        lower = torch.where(move_lower, midpoint, lower)
        upper = torch.where(move_lower, upper, midpoint)

    log_ratios = torch.minimum(log_ratio_cap, upper + supported_logits)
    weights = torch.zeros_like(working)
    weights[support] = supported_base * log_ratios.exp()
    return weights.to(dtype=scores.dtype).detach()


def robust_risk(losses: Tensor, weights: Tensor) -> Tensor:
    """Form a weighted risk while treating controller weights as a dual state."""

    if losses.ndim != 1 or losses.shape != weights.shape:
        raise ValueError("losses and weights must be equal-length vectors")
    if torch.any(weights < 0) or not torch.all(torch.isfinite(weights)):
        raise ValueError("weights must be finite and non-negative")
    if not torch.isclose(weights.sum(), weights.new_tensor(1.0)):
        raise ValueError("weights must sum to one")
    return torch.dot(weights.detach(), losses)
