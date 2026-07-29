import torch

from ssilite.adversarial import (
    EnvironmentAdversary,
    EnvironmentAdversaryConfig,
    equal_environment_weights,
)


def test_equal_environment_weights_separate_partition_from_ranking() -> None:
    assignments = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    trust = torch.tensor([1.0, 0.5, 0.5, 1.0], dtype=torch.float64)

    weights = equal_environment_weights(assignments, trust)

    torch.testing.assert_close(weights.sum(), weights.new_tensor(1.0))
    torch.testing.assert_close(weights[assignments == 0].sum(), weights.new_tensor(0.5))
    torch.testing.assert_close(weights[assignments == 1].sum(), weights.new_tensor(0.5))
    torch.testing.assert_close(
        weights,
        torch.tensor([0.25, 0.125, 0.125, 0.5], dtype=trust.dtype),
    )


def test_environment_adversary_upweights_high_risk_group_not_single_outlier() -> None:
    assignments = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    trust = torch.ones(4, dtype=torch.float64)
    controller = EnvironmentAdversary(
        assignments,
        trust,
        EnvironmentAdversaryConfig(
            tail_fraction=0.5,
            temperature=0.05,
            dual_step=1.0,
        ),
    )

    allocation = controller.update(
        torch.tensor([0.1, 0.2, 0.8, 1.0], dtype=torch.float64)
    )

    torch.testing.assert_close(
        allocation.environment_risks,
        torch.tensor([0.15, 0.9], dtype=torch.float64),
    )
    assert allocation.environment_weights[1] > allocation.environment_weights[0]
    torch.testing.assert_close(
        allocation.example_weights[assignments == 1].sum(),
        allocation.environment_weights[1],
    )
    torch.testing.assert_close(
        allocation.example_weights[assignments == 0].sum(),
        allocation.environment_weights[0],
    )


def test_environment_adversary_is_permutation_equivariant() -> None:
    assignments = torch.tensor([0, 1, 0, 2, 1, 2], dtype=torch.long)
    trust = torch.tensor([1.0, 0.4, 0.8, 0.7, 1.0, 0.5])
    losses = torch.tensor([0.2, 1.2, 0.4, 0.8, 0.6, 0.9])
    config = EnvironmentAdversaryConfig(dual_step=0.4)
    original = EnvironmentAdversary(assignments, trust, config).update(losses)

    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    permuted = EnvironmentAdversary(
        assignments[permutation],
        trust[permutation],
        config,
    ).update(losses[permutation])

    torch.testing.assert_close(
        permuted.example_weights,
        original.example_weights[permutation],
    )
    torch.testing.assert_close(
        permuted.environment_weights,
        original.environment_weights,
    )
