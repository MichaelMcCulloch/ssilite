"""Small model and evaluation helpers for the synthetic experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .data import DatasetSplit


class MechanismMLP(nn.Module):
    """A small nonlinear classifier capable of context-dependent mechanisms."""

    def __init__(self, input_dimensions: int, hidden_dimensions: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimensions, hidden_dimensions),
            nn.Tanh(),
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.Tanh(),
            nn.Linear(hidden_dimensions, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class Accuracy:
    overall: float
    majority: float
    minority: float


@torch.no_grad()
def accuracy_by_group(model: nn.Module, split: DatasetSplit) -> Accuracy:
    prediction = model(split.features) >= 0
    target = split.clean_labels.to(dtype=torch.bool)
    correct = prediction == target

    def group_mean(mask: Tensor) -> float:
        if not torch.any(mask):
            return float("nan")
        return float(correct[mask].to(dtype=torch.float32).mean().item())

    return Accuracy(
        overall=float(correct.to(dtype=torch.float32).mean().item()),
        majority=group_mean(~split.minority),
        minority=group_mean(split.minority),
    )
