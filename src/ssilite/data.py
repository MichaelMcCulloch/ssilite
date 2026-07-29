"""Synthetic support-limited classification problem."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DatasetSplit:
    features: Tensor
    labels: Tensor
    clean_labels: Tensor
    minority: Tensor
    flipped: Tensor

    def take(self, indices: Tensor) -> DatasetSplit:
        return DatasetSplit(
            features=self.features[indices],
            labels=self.labels[indices],
            clean_labels=self.clean_labels[indices],
            minority=self.minority[indices],
            flipped=self.flipped[indices],
        )


@dataclass(frozen=True)
class SupportProblem:
    train: DatasetSplit
    reservoir: DatasetSplit
    test: DatasetSplit


def _orthogonal_mechanisms(
    dimensions: int, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    first = torch.randn(dimensions, generator=generator)
    first = first / first.norm()
    second = torch.randn(dimensions, generator=generator)
    second = second - torch.dot(second, first) * first
    second = second / second.norm()
    return first, second


def _make_split(
    count: int,
    first: Tensor,
    second: Tensor,
    *,
    minority_fraction: float,
    label_noise: float,
    context_separation: float,
    generator: torch.Generator,
) -> DatasetSplit:
    core = torch.randn(count, first.numel(), generator=generator)
    minority = torch.rand(count, generator=generator) < minority_fraction
    first_logit = core @ first
    second_logit = core @ second
    logits = torch.where(minority, second_logit, first_logit)
    clean_labels = (logits >= 0).to(dtype=torch.float32)

    context = 0.35 * torch.randn(count, 2, generator=generator)
    direction = minority.to(dtype=context.dtype).mul(2).sub(1)
    context[:, 0] += context_separation * direction
    context[:, 1] += 0.5 * direction
    features = torch.cat((core, context), dim=1)

    flipped = torch.rand(count, generator=generator) < label_noise
    labels = torch.where(flipped, 1 - clean_labels, clean_labels)
    return DatasetSplit(
        features=features,
        labels=labels,
        clean_labels=clean_labels,
        minority=minority,
        flipped=flipped,
    )


def make_support_problem(
    *,
    train_size: int = 512,
    reservoir_size: int = 4096,
    test_size: int = 4096,
    core_dimensions: int = 18,
    minority_fraction: float = 0.05,
    label_noise: float = 0.04,
    context_separation: float = 3.0,
    seed: int = 0,
) -> SupportProblem:
    """Create a rare, context-identifiable mechanism plus label corruption."""

    if min(train_size, reservoir_size, test_size, core_dimensions) <= 0:
        raise ValueError("all sizes and core_dimensions must be positive")
    if not 0 < minority_fraction < 1:
        raise ValueError("minority_fraction must lie strictly between zero and one")
    if not 0 <= label_noise < 0.5:
        raise ValueError("label_noise must lie in [0, 0.5)")

    generator = torch.Generator().manual_seed(seed)
    first, second = _orthogonal_mechanisms(core_dimensions, generator)
    train = _make_split(
        train_size,
        first,
        second,
        minority_fraction=minority_fraction,
        label_noise=label_noise,
        context_separation=context_separation,
        generator=generator,
    )
    reservoir = _make_split(
        reservoir_size,
        first,
        second,
        minority_fraction=minority_fraction,
        label_noise=label_noise,
        context_separation=context_separation,
        generator=generator,
    )
    test = _make_split(
        test_size,
        first,
        second,
        minority_fraction=minority_fraction,
        label_noise=0.0,
        context_separation=context_separation,
        generator=generator,
    )
    return SupportProblem(train=train, reservoir=reservoir, test=test)
