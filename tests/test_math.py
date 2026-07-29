import math

import pytest
import torch

from ssilite.allocation import (
    greedy_precision_allocation,
    variance_aware_sampling_probabilities,
)
from ssilite.controller import ControllerConfig, JointController
from ssilite.risk import capped_entropic_weights, empirical_cvar_weights


def test_empirical_cvar_solves_capped_simplex_with_fractional_tail() -> None:
    scores = torch.arange(5, dtype=torch.float64)
    weights = empirical_cvar_weights(scores, alpha=0.3)
    expected = torch.tensor([0, 0, 0, 1 / 3, 2 / 3], dtype=torch.float64)

    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=weights.dtype))
    assert weights.max() <= 1 / (0.3 * scores.numel())
    torch.testing.assert_close(
        weights @ scores, torch.tensor(11 / 3, dtype=weights.dtype)
    )


def test_entropic_weights_are_capped_shift_and_permutation_equivariant() -> None:
    scores = torch.tensor([0.0, 0.0, math.log(2)], dtype=torch.float64)
    expected = torch.tensor([0.25, 0.25, 0.5], dtype=torch.float64)
    weights = capped_entropic_weights(scores, alpha=0.5, temperature=1.0)
    torch.testing.assert_close(weights, expected, rtol=1e-12, atol=1e-12)

    permutation = torch.tensor([2, 0, 1])
    permuted = capped_entropic_weights(
        scores[permutation] + 91, alpha=0.5, temperature=1.0
    )
    torch.testing.assert_close(permuted, weights[permutation])


def test_entropic_weights_remain_correct_across_a_wide_score_range() -> None:
    scores = torch.tensor([0.0, -1.0, -2.0, -100.0], dtype=torch.float64)
    weights = capped_entropic_weights(scores, alpha=0.625, temperature=1.0)

    torch.testing.assert_close(
        weights,
        torch.tensor([0.4, 0.4, 0.2, 0.0], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_variance_aware_sampling_attains_the_unconstrained_optimum() -> None:
    robust_weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    second_moments = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
    probabilities = variance_aware_sampling_probabilities(
        robust_weights,
        second_moments,
        defensive_mass=0,
        exploration=0,
    )

    expected = torch.tensor([5 / 17, 6 / 17, 6 / 17], dtype=torch.float64)
    torch.testing.assert_close(probabilities, expected)
    actual_second_moment = (
        robust_weights.square() * second_moments / probabilities
    ).sum()
    optimum = (robust_weights * second_moments.sqrt()).sum().square()
    torch.testing.assert_close(actual_second_moment, optimum)


def test_defensive_sampling_bounds_importance_ratios_and_explores() -> None:
    robust_weights = torch.tensor([0.8, 0.2, 0.0], dtype=torch.float64)
    probabilities = variance_aware_sampling_probabilities(
        robust_weights,
        torch.tensor([1.0, 100.0, 0.0], dtype=torch.float64),
        defensive_mass=0.2,
        exploration=0.06,
    )

    assert torch.all(probabilities >= 0.06 / 3 - 1e-12)
    assert torch.all(robust_weights / probabilities <= 5 + 1e-12)
    torch.testing.assert_close(
        probabilities.sum(), torch.tensor(1.0, dtype=probabilities.dtype)
    )


def test_importance_estimator_is_unbiased_by_enumeration() -> None:
    robust_weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    probabilities = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    gradients = torch.tensor(
        [[1.0, -2.0], [3.0, 0.5], [-4.0, 2.0]], dtype=torch.float64
    )

    conditional_estimates = (robust_weights / probabilities).unsqueeze(1) * gradients
    actual = (probabilities.unsqueeze(1) * conditional_estimates).sum(dim=0)
    expected = (robust_weights.unsqueeze(1) * gradients).sum(dim=0)
    torch.testing.assert_close(actual, expected)


def test_precision_allocation_respects_expected_budget() -> None:
    robust_weights = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    probabilities = torch.full((3,), 1 / 3, dtype=torch.float64)
    variances = torch.tensor(
        [[1.0, 0.01], [1.0, 0.01], [1.0, 0.01]], dtype=torch.float64
    )
    allocation = greedy_precision_allocation(
        robust_weights,
        probabilities,
        variances,
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        (4, 8),
        mean_cost_budget=4 / 3,
    )

    torch.testing.assert_close(
        allocation.bits, torch.tensor([8, 4, 4], dtype=torch.long)
    )
    assert allocation.expected_cost <= 4 / 3 + 1e-12
    all_low_variance = (robust_weights.square() / probabilities * variances[:, 0]).sum()
    assert allocation.variance_term < all_low_variance


def test_precision_allocation_promotes_integer_costs_and_rejects_zero_p() -> None:
    robust_weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    probabilities = torch.tensor([0.5, 0.5], dtype=torch.float64)
    variances = torch.tensor([[1.0, 0.1], [1.0, 0.1]], dtype=torch.float64)

    allocation = greedy_precision_allocation(
        robust_weights,
        probabilities,
        variances,
        torch.tensor([1, 2]),
        (4, 8),
        mean_cost_budget=1.5,
    )
    assert allocation.expected_cost <= 1.5

    with pytest.raises(ValueError, match="strictly positive"):
        greedy_precision_allocation(
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            variances,
            torch.tensor([1, 2]),
            (4, 8),
            mean_cost_budget=1.5,
        )

    with pytest.raises(ValueError, match="finite"):
        greedy_precision_allocation(
            robust_weights,
            probabilities,
            variances,
            torch.tensor([1, 2]),
            (4, 8),
            mean_cost_budget=float("nan"),
        )


def test_joint_controller_keeps_q_p_and_precision_separate() -> None:
    controller = JointController(
        ControllerConfig(
            tail_fraction=0.5,
            dual_step=1.0,
            score_bits=8,
            allocation_rounds=2,
        )
    )
    scores = torch.tensor([0.1, 0.2, 0.8, 1.0], dtype=torch.float64)
    proxy = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    allocation = controller.allocate(
        scores,
        proxy,
        batch_size=16,
        generator=torch.Generator().manual_seed(3),
    )

    torch.testing.assert_close(
        allocation.robust_weights.sum(),
        torch.tensor(1.0, dtype=allocation.robust_weights.dtype),
    )
    torch.testing.assert_close(
        allocation.sampling_probabilities.sum(),
        torch.tensor(1.0, dtype=allocation.sampling_probabilities.dtype),
    )
    assert allocation.robust_weights.max() <= 0.5 + 1e-12
    assert allocation.expected_precision_cost <= 2.0 + 1e-12
    torch.testing.assert_close(
        allocation.importance_weights,
        allocation.robust_weights[allocation.indices]
        / allocation.sampling_probabilities[allocation.indices],
    )


def test_known_irreducible_loss_filters_a_corrupt_tail() -> None:
    targets = torch.cat(
        (
            torch.ones(48, dtype=torch.float64),
            torch.full((16,), -4.0, dtype=torch.float64),
        )
    )
    irreducible = 0.5 * (1.0 - targets).square()

    def optimize(shaped: bool) -> float:
        parameter = torch.tensor(0.0, dtype=torch.float64)
        for _ in range(80):
            losses = 0.5 * (parameter - targets).square()
            scores = losses - irreducible if shaped else losses
            weights = empirical_cvar_weights(scores, alpha=0.25)
            gradient = torch.dot(weights, parameter - targets)
            parameter -= 0.1 * gradient
        return float(parameter.item())

    raw = optimize(shaped=False)
    shaped = optimize(shaped=True)
    assert shaped > 0.999
    assert 0.5 * (shaped - 1) ** 2 < 1e-6
    assert 0.5 * (raw - 1) ** 2 > 2
