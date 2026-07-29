"""Support-changing acquisition policies.

Reweighting a finite training set cannot manufacture examples from a missing
subpopulation.  This module therefore keeps acquisition separate from the
within-support robust objective and gradient estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Acquisition:
    """Indices selected from a reservoir and diagnostics about their coverage."""

    indices: Tensor
    labeled_counts_before: Tensor
    acquired_counts: Tensor
    centers: Tensor


def _standardize(features: Tensor) -> Tensor:
    mean = features.mean(dim=0, keepdim=True)
    scale = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (features - mean) / scale


def kmeans(
    features: Tensor,
    num_clusters: int,
    *,
    iterations: int = 30,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Cluster rows of ``features`` with a small, dependency-free k-means."""

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty matrix")
    if not 1 <= num_clusters <= features.shape[0]:
        raise ValueError("num_clusters must be between 1 and the number of rows")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not torch.all(torch.isfinite(features)):
        raise ValueError("features must be finite")

    first = torch.randint(
        features.shape[0],
        (),
        device=features.device,
        generator=generator,
    )
    chosen = [int(first.item())]
    nearest_sq = torch.cdist(features, features[first].unsqueeze(0)).square().squeeze(1)
    for _ in range(1, num_clusters):
        total = nearest_sq.sum()
        if total <= 0:
            remaining = torch.tensor(
                [i for i in range(features.shape[0]) if i not in chosen],
                device=features.device,
            )
            next_index = remaining[
                torch.randint(
                    remaining.numel(),
                    (),
                    device=features.device,
                    generator=generator,
                )
            ]
        else:
            next_index = torch.multinomial(
                nearest_sq / total, 1, generator=generator
            ).squeeze(0)
        chosen.append(int(next_index.item()))
        distance_sq = (
            torch.cdist(features, features[next_index].unsqueeze(0)).square().squeeze(1)
        )
        nearest_sq = torch.minimum(nearest_sq, distance_sq)

    centers = features[torch.tensor(chosen, device=features.device)].clone()
    assignments = torch.zeros(
        features.shape[0], dtype=torch.long, device=features.device
    )
    for _ in range(iterations):
        distances = torch.cdist(features, centers)
        new_assignments = distances.argmin(dim=1)
        new_centers = centers.clone()
        for cluster in range(num_clusters):
            members = features[new_assignments == cluster]
            if members.numel() > 0:
                new_centers[cluster] = members.mean(dim=0)
        converged = torch.equal(assignments, new_assignments)
        assignments, centers = new_assignments, new_centers
        if converged:
            break
    return assignments, centers


def acquire_for_cluster_coverage(
    labeled_features: Tensor,
    reservoir_features: Tensor,
    count: int,
    *,
    num_clusters: int = 2,
    iterations: int = 30,
    generator: torch.Generator | None = None,
) -> Acquisition:
    """Acquire representative points from under-covered input-space clusters.

    Clusters are discovered without labels from the union of the current
    support and reservoir.  The policy repeatedly selects a representative
    point from the cluster with the smallest labeled count.  This is an
    intentionally simple example of exogenous support information: it uses raw
    observations rather than a loss signal produced by the biased model.
    """

    if labeled_features.ndim != 2 or reservoir_features.ndim != 2:
        raise ValueError("labeled_features and reservoir_features must be matrices")
    if labeled_features.shape[1] != reservoir_features.shape[1]:
        raise ValueError("labeled and reservoir features must have equal width")
    if not 0 <= count <= reservoir_features.shape[0]:
        raise ValueError("count must fit within the reservoir")
    if labeled_features.shape[0] == 0:
        raise ValueError("at least one labeled example is required")

    combined = torch.cat((labeled_features, reservoir_features), dim=0)
    normalized = _standardize(combined)
    assignments, centers = kmeans(
        normalized,
        num_clusters,
        iterations=iterations,
        generator=generator,
    )
    split = labeled_features.shape[0]
    labeled_assignments = assignments[:split]
    reservoir_assignments = assignments[split:]
    labeled_counts = torch.bincount(labeled_assignments, minlength=num_clusters).to(
        dtype=torch.long
    )
    running_counts = labeled_counts.clone()
    available = torch.ones(
        reservoir_features.shape[0],
        dtype=torch.bool,
        device=reservoir_features.device,
    )
    selected: list[Tensor] = []
    acquired_counts = torch.zeros_like(labeled_counts)

    for _ in range(count):
        candidates_by_cluster = torch.bincount(
            reservoir_assignments[available], minlength=num_clusters
        )
        eligible = candidates_by_cluster > 0
        priority = running_counts.to(dtype=normalized.dtype)
        priority = torch.where(eligible, priority, torch.full_like(priority, torch.inf))
        cluster = int(priority.argmin().item())

        candidate_indices = torch.nonzero(
            available & (reservoir_assignments == cluster), as_tuple=False
        ).squeeze(1)
        candidate_rows = normalized[split + candidate_indices]
        representative_position = (
            torch.cdist(candidate_rows, centers[cluster].unsqueeze(0))
            .squeeze(1)
            .argmin()
        )
        index = candidate_indices[representative_position]
        selected.append(index)
        available[index] = False
        running_counts[cluster] += 1
        acquired_counts[cluster] += 1

    indices = (
        torch.stack(selected)
        if selected
        else torch.empty(0, dtype=torch.long, device=reservoir_features.device)
    )
    return Acquisition(
        indices=indices,
        labeled_counts_before=labeled_counts,
        acquired_counts=acquired_counts,
        centers=centers,
    )
