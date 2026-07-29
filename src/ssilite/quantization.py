"""Quantization primitives used by the prototype estimator.

The project deliberately emulates per-example precision.  Stock PyTorch kernels
cannot execute individual examples from one batch at different arithmetic
formats, while an emulation lets us test the controller's statistical claims
without conflating them with a custom-kernel implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _quantization_levels(bits: int) -> int:
    if bits < 2:
        raise ValueError(f"bits must be at least 2, got {bits}")
    return (1 << (bits - 1)) - 1


def quantization_variance_proxy(magnitude_sq: Tensor, bits: Tensor | int) -> Tensor:
    """Return a relative stochastic-quantization variance proxy.

    For a symmetric uniform quantizer, the squared bin width scales as
    ``magnitude_sq / levels**2``.  The omitted constant is shared by all
    precision choices, so it does not affect the allocator's ordering.
    """

    if torch.is_tensor(bits):
        if torch.any(bits < 2):
            raise ValueError("all precision values must be at least 2 bits")
        levels = (
            torch.pow(
                magnitude_sq.new_tensor(2.0), bits.to(dtype=magnitude_sq.dtype) - 1
            )
            - 1
        )
    else:
        levels = magnitude_sq.new_tensor(float(_quantization_levels(bits)))
    return magnitude_sq.clamp_min(0) / levels.square().clamp_min(
        torch.finfo(magnitude_sq.dtype).tiny
    )


def deterministic_quantize(values: Tensor, bits: int) -> Tensor:
    """Round a tensor to a symmetric ``bits``-wide grid.

    This is used only for score/ranking emulation.  Gradient quantization uses
    :func:`stochastic_quantize` so that the gradient estimator remains
    conditionally unbiased.
    """

    levels = _quantization_levels(bits)
    if bits >= 32 or values.numel() == 0:
        return values.clone()
    max_abs = values.detach().abs().amax()
    if not torch.isfinite(max_abs):
        raise ValueError("values must be finite")
    if max_abs == 0:
        return values.clone()
    scale = max_abs / levels
    return torch.round(values / scale).clamp(-levels, levels) * scale


def stochastic_quantize(
    values: Tensor,
    bits: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Unbiasedly quantize ``values`` on a per-tensor symmetric grid.

    Values are stochastically rounded between adjacent grid points.  Conditional
    on the input tensor and its scale, the returned tensor has expectation equal
    to ``values``.  A 32-bit request is treated as an exact pass-through.
    """

    levels = _quantization_levels(bits)
    if bits >= 32 or values.numel() == 0:
        return values.clone()

    detached = values.detach()
    max_abs = detached.abs().amax()
    if not torch.isfinite(max_abs):
        raise ValueError("values must be finite")
    if max_abs == 0:
        return values.clone()

    scale = max_abs / levels
    normalized = (values.abs() / scale).clamp_max(float(levels))
    lower = normalized.floor()
    probability_up = normalized - lower
    draw = torch.rand(
        probability_up.shape,
        dtype=probability_up.dtype,
        device=probability_up.device,
        generator=generator,
    )
    rounded = lower + (draw < probability_up).to(values.dtype)
    return values.sign() * rounded * scale


def stochastic_quantize_rows(
    values: Tensor,
    bits: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Unbiasedly quantize each leading-dimension row at its own precision."""

    if values.ndim < 1 or bits.ndim != 1 or values.shape[0] != bits.numel():
        raise ValueError("bits must provide one precision per leading-dimension row")
    if torch.any(bits < 2):
        raise ValueError("all precision values must be at least 2 bits")
    if not torch.all(torch.isfinite(values)):
        raise ValueError("values must be finite")

    flattened = values.reshape(values.shape[0], -1)
    levels = (
        torch.pow(
            flattened.new_tensor(2.0),
            bits.to(device=values.device, dtype=values.dtype) - 1,
        )
        - 1
    )
    max_abs = flattened.detach().abs().amax(dim=1)
    scales = max_abs / levels
    safe_scales = scales.clamp_min(torch.finfo(values.dtype).tiny)
    normalized = (flattened.abs() / safe_scales.unsqueeze(1)).clamp_max(
        levels.unsqueeze(1)
    )
    lower = normalized.floor()
    probability_up = normalized - lower
    draw = torch.rand(
        probability_up.shape,
        dtype=probability_up.dtype,
        device=probability_up.device,
        generator=generator,
    )
    rounded = lower + (draw < probability_up).to(values.dtype)
    quantized = (
        flattened.sign()
        * rounded
        * torch.where(max_abs > 0, scales, torch.zeros_like(scales)).unsqueeze(1)
    )
    exact_rows = bits >= 32
    quantized[exact_rows] = flattened[exact_rows]
    return quantized.reshape_as(values)
