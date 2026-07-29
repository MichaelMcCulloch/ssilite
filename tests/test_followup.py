import torch

from ssilite.bootstrap import BootstrapConfig
from ssilite.followup import _binary_auroc, _checkpoint_count


def test_binary_auroc_is_tie_correct() -> None:
    scores = torch.tensor([0.0, 0.5, 0.5, 1.0])
    positives = torch.tensor([False, True, False, True])

    assert _binary_auroc(scores, positives) == 0.875
    assert _binary_auroc(torch.ones(4), positives) == 0.5


def test_checkpoint_count_includes_endpoints_without_duplicates() -> None:
    config = BootstrapConfig(
        folds=2,
        repeats=2,
        rounds=1,
        training_steps=2,
        checkpoints=5,
    )

    assert _checkpoint_count(config) == 3
