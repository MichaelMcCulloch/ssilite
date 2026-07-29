"""Importance-weighted, mixed-precision gradient estimation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap

from .quantization import stochastic_quantize, stochastic_quantize_rows


@dataclass(frozen=True)
class GradientEstimate:
    """Diagnostics from a quantized per-example gradient estimate."""

    mean_grad_norm: float
    quantization_mse: float
    max_importance_weight: float


def apply_importance_weighted_gradients(
    model: nn.Module,
    per_sample_losses: Tensor,
    importance_weights: Tensor,
    precision_bits: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> GradientEstimate:
    """Populate ``parameter.grad`` with a quantized importance estimator.

    If examples are sampled from ``p`` and ``importance_weights = q / p``, the
    unquantized estimator is unbiased for ``sum_i q_i grad(loss_i)``.  The
    stochastic quantizer preserves this conditional expectation.

    This reference implementation computes per-example gradients explicitly.
    It is intentionally simple and is not presented as an efficient training
    kernel.
    """

    if per_sample_losses.ndim != 1:
        raise ValueError("per_sample_losses must be one-dimensional")
    batch_size = per_sample_losses.numel()
    if batch_size == 0:
        raise ValueError("cannot estimate a gradient from an empty batch")
    if importance_weights.shape != per_sample_losses.shape:
        raise ValueError("importance_weights must match per_sample_losses")
    if precision_bits.shape != per_sample_losses.shape:
        raise ValueError("precision_bits must match per_sample_losses")
    if not torch.all(torch.isfinite(per_sample_losses)):
        raise ValueError("per_sample_losses must be finite")
    if not torch.all(torch.isfinite(importance_weights)):
        raise ValueError("importance_weights must be finite")
    if torch.any(importance_weights < 0):
        raise ValueError("importance_weights must be non-negative")

    parameters: Sequence[nn.Parameter] = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("model has no trainable parameters")

    accumulated = [torch.zeros_like(parameter) for parameter in parameters]
    exact_sq_norm = per_sample_losses.new_zeros(())
    error_sq = per_sample_losses.new_zeros(())

    for sample_index in range(batch_size):
        gradients = torch.autograd.grad(
            per_sample_losses[sample_index],
            parameters,
            retain_graph=sample_index + 1 < batch_size,
            allow_unused=True,
        )
        bit_width = int(precision_bits[sample_index].item())
        weight = importance_weights[sample_index] / batch_size

        sample_sq_norm = per_sample_losses.new_zeros(())
        sample_error_sq = per_sample_losses.new_zeros(())
        for position, gradient in enumerate(gradients):
            if gradient is None:
                continue
            quantized = stochastic_quantize(gradient, bit_width, generator=generator)
            accumulated[position].add_(quantized * weight)
            sample_sq_norm = sample_sq_norm + gradient.detach().square().sum()
            sample_error_sq = (
                sample_error_sq
                + (quantized.detach() - gradient.detach()).square().sum()
            )
        exact_sq_norm = exact_sq_norm + sample_sq_norm
        error_sq = error_sq + sample_error_sq

    for parameter, gradient in zip(parameters, accumulated, strict=True):
        parameter.grad = gradient

    return GradientEstimate(
        mean_grad_norm=float((exact_sq_norm / batch_size).sqrt().item()),
        quantization_mse=float((error_sq / batch_size).item()),
        max_importance_weight=float(importance_weights.max().item()),
    )


def apply_batched_binary_gradients(
    model: nn.Module,
    features: Tensor,
    targets: Tensor,
    importance_weights: Tensor,
    precision_bits: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> GradientEstimate:
    """Vectorized binary-classification version of the reference estimator."""

    if features.ndim < 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty batch")
    batch_size = features.shape[0]
    expected_shape = (batch_size,)
    if targets.shape != expected_shape:
        raise ValueError("targets must contain one scalar per example")
    if importance_weights.shape != expected_shape:
        raise ValueError("importance_weights must contain one value per example")
    if precision_bits.shape != expected_shape:
        raise ValueError("precision_bits must contain one value per example")
    if torch.any(importance_weights < 0) or not torch.all(
        torch.isfinite(importance_weights)
    ):
        raise ValueError("importance_weights must be finite and non-negative")

    parameters: dict[str, Tensor] = dict(model.named_parameters())
    buffers: dict[str, Tensor] = dict(model.named_buffers())

    def sample_loss(
        current_parameters: dict[str, Tensor],
        current_buffers: dict[str, Tensor],
        sample: Tensor,
        target: Tensor,
    ) -> Tensor:
        logit = functional_call(
            model,
            (current_parameters, current_buffers),
            (sample.unsqueeze(0),),
        ).squeeze()
        return F.binary_cross_entropy_with_logits(logit, target)

    gradient_function = grad(sample_loss)
    per_sample_gradients = vmap(
        gradient_function,
        in_dims=(None, None, 0, 0),
    )(parameters, buffers, features, targets)

    exact_sq_norm = features.new_zeros(())
    error_sq = features.new_zeros(())
    for name, parameter in model.named_parameters():
        gradient_rows = per_sample_gradients[name]
        quantized_rows = stochastic_quantize_rows(
            gradient_rows,
            precision_bits,
            generator=generator,
        )
        weight_shape = (batch_size,) + (1,) * (gradient_rows.ndim - 1)
        parameter.grad = (quantized_rows * importance_weights.view(weight_shape)).mean(
            dim=0
        )
        exact_sq_norm += gradient_rows.detach().square().sum()
        error_sq += (quantized_rows.detach() - gradient_rows.detach()).square().sum()

    return GradientEstimate(
        mean_grad_norm=float((exact_sq_norm / batch_size).sqrt().item()),
        quantization_mse=float((error_sq / batch_size).item()),
        max_importance_weight=float(importance_weights.max().item()),
    )
