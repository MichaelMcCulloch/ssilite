#!/usr/bin/env python
"""Plot uncertainty-spawning causal arms and write normalized artifacts.

Example:

    uv run ssilite-uncertainty-spawning \
      --seeds 0 1 2 3 4 5 6 7 \
      --budgets 512 1024 1536 --device cuda |
    uv run --with 'matplotlib>=3.10' python \
      scripts/plot_uncertainty_spawning.py \
      --output-prefix artifacts/uncertainty_spawning
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArmStyle:
    field: str
    label: str
    color: str
    linestyle: str
    marker: str


ARMS = (
    ArmStyle("single", "Single expert", "#202020", "-", "o"),
    ArmStyle("raw_loss", "Raw-loss spawner", "#D55E00", "--", "v"),
    ArmStyle("expected_only", "Expected only", "#777777", ":", "o"),
    ArmStyle("unvalidated", "Unvalidated spawner", "#E69F00", "-.", "D"),
    ArmStyle("joint", "Joint controller", "#009E73", "-", "s"),
    ArmStyle("environment_oracle", "Environment oracle", "#0072B2", "--", "^"),
)


def _load_results(path: str) -> list[dict[str, Any]]:
    if path == "-":
        payload = json.load(sys.stdin)
    else:
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
    if not isinstance(payload, list) or not payload:
        raise ValueError("input must be a non-empty JSON result list")
    if not all(isinstance(result, dict) for result in payload):
        raise ValueError("every result must be a JSON object")
    return payload


def _validated_axes(
    results: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    seeds = [int(result["seed"]) for result in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError("result seeds must be unique")
    reference_budgets = [int(point["total_labels"]) for point in results[0]["points"]]
    if not reference_budgets:
        raise ValueError("every result must contain benchmark points")
    expected_arms = {style.field for style in ARMS}
    for result in results:
        budgets = [int(point["total_labels"]) for point in result["points"]]
        if budgets != reference_budgets:
            raise ValueError("all seeds must share identical support budgets")
        for point in result["points"]:
            if set(point["arms"]) != expected_arms:
                raise ValueError("every point must contain all causal arms")
    return seeds, reference_budgets


def _interval(values: list[float]) -> tuple[float, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("plotted values must be finite and non-empty")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return mean, 1.96 * standard_error


def _summary_rows(
    results: list[dict[str, Any]],
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for result in results:
        for point in result["points"]:
            for style in ARMS:
                arm = point["arms"][style.field]
                compute = arm["compute"]
                forward_work = sum(
                    int(value)
                    for key, value in compute.items()
                    if "forward" in key or key == "sparse_inference_examples"
                )
                backward_work = sum(
                    int(value) for key, value in compute.items() if "backward" in key
                )
                rows.append(
                    {
                        "seed": int(result["seed"]),
                        "device": str(result["device"]),
                        "total_labels": int(point["total_labels"]),
                        "arm": style.field,
                        "overall_accuracy": float(arm["accuracy"]["overall"]),
                        "common_accuracy": float(arm["accuracy"]["common"]),
                        "rare_rule_accuracy": float(arm["accuracy"]["rare_rule"]),
                        "stochastic_pocket_accuracy": float(
                            arm["accuracy"]["stochastic_pocket"]
                        ),
                        "active_experts": int(arm["active_experts"]),
                        "birth_count": int(arm["birth_count"]),
                        "rare_rule_births": int(arm["rare_rule_births"]),
                        "stochastic_false_births": int(arm["stochastic_false_births"]),
                        "surprise_rate": float(arm["surprise_rate"]),
                        "proposal_count": int(arm["proposal_count"]),
                        "stochastic_rejections": int(arm["stochastic_rejections"]),
                        "expected_uncertainty_expert_0": (
                            ""
                            if not arm["expected_uncertainties"]
                            else float(arm["expected_uncertainties"][0])
                        ),
                        "rare_switch_margin_mean": (
                            ""
                            if arm["rare_evidence_mean"] is None
                            else float(arm["rare_evidence_mean"])
                        ),
                        "stochastic_switch_margin_mean": (
                            ""
                            if arm["stochastic_evidence_mean"] is None
                            else float(arm["stochastic_evidence_mean"])
                        ),
                        "forward_work": forward_work,
                        "backward_work": backward_work,
                    }
                )
    return rows


def _series(
    results: list[dict[str, Any]],
    *,
    arm: str,
    field: str,
) -> tuple[list[float], list[float]]:
    means: list[float] = []
    half_widths: list[float] = []
    point_count = len(results[0]["points"])
    for point_index in range(point_count):
        values = [
            float(result["points"][point_index]["arms"][arm][field])
            if field
            not in {
                "common",
                "rare_rule",
                "stochastic_pocket",
            }
            else float(result["points"][point_index]["arms"][arm]["accuracy"][field])
            for result in results
        ]
        mean, half_width = _interval(values)
        means.append(mean)
        half_widths.append(half_width)
    return means, half_widths


def _plot(
    results: list[dict[str, Any]],
    budgets: list[int],
    output_prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True)
    panels = (
        ("common", "Common deterministic accuracy", (0.45, 1.01)),
        ("rare_rule", "Rare alternate-rule accuracy", (0.45, 1.01)),
        ("stochastic_false_births", "Stochastic-pocket false births", (-0.05, 2.1)),
        ("rare_rule_births", "Rare-rule specialist births", (-0.05, 2.1)),
    )
    for axis, (field, title, limits) in zip(axes.flat, panels, strict=True):
        for style in ARMS:
            means, half_widths = _series(
                results,
                arm=style.field,
                field=field,
            )
            axis.plot(
                budgets,
                means,
                label=style.label,
                color=style.color,
                linestyle=style.linestyle,
                marker=style.marker,
                linewidth=2,
            )
            if len(results) > 1:
                axis.fill_between(
                    budgets,
                    [
                        mean - half_width
                        for mean, half_width in zip(
                            means,
                            half_widths,
                            strict=True,
                        )
                    ],
                    [
                        mean + half_width
                        for mean, half_width in zip(
                            means,
                            half_widths,
                            strict=True,
                        )
                    ],
                    color=style.color,
                    alpha=0.10,
                    linewidth=0,
                )
        axis.set_title(title)
        axis.set_ylim(*limits)
        axis.grid(alpha=0.25)
    for axis in axes[1]:
        axis.set_xlabel("Observed labels")
    axes[0, 0].set_ylabel("Clean accuracy")
    axes[1, 0].set_ylabel("Mean births per seed")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Expected vs. unexpected uncertainty: held-out expert spawning",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(
        output_prefix.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def _write_artifacts(
    results: list[dict[str, Any]],
    rows: list[dict[str, str | int | float]],
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with output_prefix.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    with output_prefix.with_suffix(".csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot all expected/unexpected-uncertainty spawning arms."
    )
    parser.add_argument(
        "--input",
        default="-",
        help="uncertainty-spawning JSON path, or '-' for stdin",
    )
    parser.add_argument(
        "--output-prefix",
        default="artifacts/uncertainty_spawning",
        help="path stem for .svg, .png, .csv, and normalized .json outputs",
    )
    arguments = parser.parse_args(argv)
    results = _load_results(arguments.input)
    _, budgets = _validated_axes(results)
    rows = _summary_rows(results)
    output_prefix = Path(arguments.output_prefix).resolve()
    _write_artifacts(results, rows, output_prefix)
    _plot(results, budgets, output_prefix)
    for suffix in (".svg", ".png", ".csv", ".json"):
        print(output_prefix.with_suffix(suffix))


if __name__ == "__main__":
    main()
