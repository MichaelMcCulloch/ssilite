import torch
from torch import nn

from ssilite.estimator import apply_batched_binary_gradients


def test_batched_binary_estimator_matches_weighted_autograd_at_32_bits() -> None:
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    features = torch.tensor([[1.0, -2.0], [-0.5, 0.25], [2.0, 1.5], [0.0, -1.0]])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    importance = torch.tensor([0.25, 1.5, 0.75, 2.0])

    logits = model(features).squeeze(1)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    expected = torch.autograd.grad(
        (importance * losses).mean(), tuple(model.parameters())
    )

    diagnostics = apply_batched_binary_gradients(
        model,
        features,
        targets,
        importance,
        torch.full((features.shape[0],), 32, dtype=torch.long),
    )

    for parameter, expected_gradient in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, expected_gradient)
    assert diagnostics.quantization_mse == 0.0
