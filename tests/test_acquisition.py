import torch

from ssilite.acquisition import acquire_for_cluster_coverage
from ssilite.data import make_support_problem


def test_coverage_acquisition_discovers_and_balances_rare_input_cluster() -> None:
    problem = make_support_problem(seed=0)

    acquisition = acquire_for_cluster_coverage(
        problem.train.features,
        problem.reservoir.features,
        count=256,
        num_clusters=4,
        generator=torch.Generator().manual_seed(17),
    )

    # The policy receives features only. Ground-truth membership is inspected
    # afterward to characterize what its label-free coverage signal discovered.
    selected_minority_rate = (
        problem.reservoir.minority[acquisition.indices].float().mean()
    )
    reservoir_minority_rate = problem.reservoir.minority.float().mean()
    final_cluster_counts = (
        acquisition.labeled_counts_before + acquisition.acquired_counts
    )

    assert selected_minority_rate >= 0.60
    assert selected_minority_rate >= 10 * reservoir_minority_rate
    assert final_cluster_counts.unique().numel() == 1
