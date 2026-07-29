"""Bounded empirical tail-risk objectives."""

from __future__ import annotations

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
    bisection_steps: int = 80,
) -> Tensor:
    """Return a smooth, capped exponential tilt of empirical scores.

    This solves the entropy-regularized CVaR inner problem.  Its solution has
    ``q_i = min(cap, c * exp(score_i / temperature))``; the scalar ``c`` is
    found by bisection.  The computation is performed in float64 for stability
    and the detached result is cast back to the input dtype.
    """

    _validate_scores(scores, alpha)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if bisection_steps < 1:
        raise ValueError("bisection_steps must be positive")

    working = scores.to(dtype=torch.float64)
    logits = (working - working.max()) / temperature
    cap = _weight_cap(scores.numel(), alpha)
    log_cap = working.new_tensor(cap).log()

    lower = -torch.logsumexp(logits, dim=0) - 2
    upper = log_cap - logits.min() + 2
    for _ in range(bisection_steps):
        midpoint = (lower + upper) / 2
        log_weights = torch.minimum(log_cap, midpoint + logits)
        total = log_weights.exp().sum()
        if total < 1:
            lower = midpoint
        else:
            upper = midpoint

    log_weights = torch.minimum(log_cap, upper + logits)
    weights = log_weights.exp()
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
