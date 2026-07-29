import pytest
import torch
from torch import nn

from ssilite.estimator import apply_importance_weighted_gradients
from ssilite.quantization import deterministic_quantize, stochastic_quantize


def test_stochastic_quantization_is_empirically_unbiased() -> None:
    values = torch.tensor([-1.0, -0.73, -0.11, 0.0, 0.19, 0.62, 1.0])
    repeated = values.repeat(20_000, 1)

    quantized = stochastic_quantize(
        repeated,
        bits=3,
        generator=torch.Generator().manual_seed(12_345),
    )

    torch.testing.assert_close(
        quantized.mean(dim=0),
        values,
        rtol=0.0,
        atol=0.006,
    )


@pytest.mark.parametrize(
    "quantize",
    [deterministic_quantize, stochastic_quantize],
)
def test_32_bit_quantization_is_exact(quantize) -> None:
    values = torch.tensor(
        [-torch.pi, -0.123456789, 0.0, 0.987654321, torch.e],
        dtype=torch.float64,
    )

    quantized = quantize(values, bits=32)

    assert torch.equal(quantized, values)
    assert quantized.data_ptr() != values.data_ptr()


def test_32_bit_importance_weighted_linear_gradient_matches_manual() -> None:
    model = nn.Linear(2, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.4, -0.7]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.2], dtype=torch.float64))

    features = torch.tensor(
        [[1.0, -2.0], [-0.5, 0.25], [2.0, 1.5], [0.0, -1.0]],
        dtype=torch.float64,
    )
    targets = torch.tensor([0.3, -0.8, 1.2, 0.5], dtype=torch.float64)
    importance_weights = torch.tensor(
        [0.25, 1.5, 0.75, 2.0],
        dtype=torch.float64,
    )
    predictions = model(features).squeeze(1)
    residuals = predictions - targets
    per_sample_losses = residuals.square()

    expected_weight_gradient = (
        2 * (importance_weights * residuals).unsqueeze(1) * features
    ).mean(dim=0, keepdim=True)
    expected_bias_gradient = (2 * importance_weights * residuals).mean().unsqueeze(0)

    diagnostics = apply_importance_weighted_gradients(
        model,
        per_sample_losses,
        importance_weights,
        torch.full((features.shape[0],), 32, dtype=torch.long),
        generator=torch.Generator().manual_seed(9),
    )

    torch.testing.assert_close(
        model.weight.grad,
        expected_weight_gradient,
        rtol=1e-14,
        atol=1e-14,
    )
    torch.testing.assert_close(
        model.bias.grad,
        expected_bias_gradient,
        rtol=1e-14,
        atol=1e-14,
    )
    assert diagnostics.quantization_mse == 0.0
    assert diagnostics.max_importance_weight == 2.0
