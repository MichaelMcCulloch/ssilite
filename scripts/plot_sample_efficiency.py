#!/usr/bin/env python
"""Plot paired formal-MoE sample-efficiency variants from ssilite JSON output.

Example:

    uv run ssilite-sample-efficiency \
      --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
      --budgets 0 32 64 128 256 --device cuda |
    uv run --with matplotlib python scripts/plot_sample_efficiency.py \
      --output-prefix artifacts/sample_efficiency_variants
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
    linestyle: str | tuple[int, tuple[int, ...]]
    marker: str
    linewidth: float
    marker_face: str | None = None


ARMS = (
    ArmStyle(
        "ordinary_mean",
        "Ordinary experts — dense mean (baseline)",
        "#202020",
        "-",
        "o",
        2.0,
    ),
    ArmStyle(
        "ordinary_routed",
        "Ordinary MoE — top-1 router (gate only)",
        "#777777",
        ":",
        "o",
        1.8,
        "none",
    ),
    ArmStyle(
        "specialist_mean",
        "Specialized experts — dense mean (no routing)",
        "#0072B2",
        "--",
        "^",
        1.8,
    ),
    ArmStyle(
        "permuted_routed_specialist",
        "Specialized MoE — permuted environments",
        "#E69F00",
        "-.",
        "D",
        1.8,
    ),
    ArmStyle(
        "routed_specialist",
        "Environment MoE — matching top-1 router",
        "#009E73",
        "-",
        "s",
        2.8,
    ),
)
BASELINE = ARMS[0].field


@dataclass(frozen=True)
class Interval:
    mean: float
    standard_deviation: float
    lower: float
    upper: float


def _t_critical_95(degrees_of_freedom: int) -> float:
    """Return a two-sided 95% Student-t critical value.

    The figure's intended run has 16 seeds (15 degrees of freedom). The
    Cornish-Fisher expansion keeps the utility dependency-free for other
    sensible seed counts and is very accurate by this sample size.
    """

    if degrees_of_freedom < 1:
        raise ValueError("at least two paired seeds are required")
    if degrees_of_freedom == 15:
        return 2.131449545559323
    z = statistics.NormalDist().inv_cdf(0.975)
    v = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4 * v)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * v**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * v**3)
    )


def _interval(values: list[float]) -> Interval:
    if len(values) < 2:
        raise ValueError("at least two seed values are required")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all plotted values must be finite")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half_width = (
        _t_critical_95(len(values) - 1) * standard_deviation / math.sqrt(len(values))
    )
    return Interval(
        mean=mean,
        standard_deviation=standard_deviation,
        lower=mean - half_width,
        upper=mean + half_width,
    )


def _load_results(input_path: str) -> list[dict[str, Any]]:
    if input_path == "-":
        payload = json.load(sys.stdin)
    else:
        with Path(input_path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("input must contain at least two seed results")
    if not all(isinstance(result, dict) for result in payload):
        raise ValueError("every seed result must be an object")
    return payload


def _validated_axes(
    results: list[dict[str, Any]],
) -> tuple[list[int], list[int], int, int, int]:
    seeds = [int(result["seed"]) for result in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed identifiers must be unique")
    first_points = results[0]["points"]
    budgets = [int(point["new_labels"]) for point in first_points]
    if budgets != sorted(set(budgets)):
        raise ValueError("budgets must be strictly increasing")
    if not budgets:
        raise ValueError("at least one budget is required")

    target = float(results[0]["config"]["minority_target"])
    majority_floor = float(results[0]["config"]["majority_floor"])
    backward_examples: int | None = None
    router_training_examples: int | None = None
    for result in results:
        if [int(point["new_labels"]) for point in result["points"]] != budgets:
            raise ValueError("all seeds must contain identical ordered budgets")
        if float(result["config"]["minority_target"]) != target:
            raise ValueError("minority targets differ across seeds")
        if float(result["config"]["majority_floor"]) != majority_floor:
            raise ValueError("majority floors differ across seeds")
        for point in result["points"]:
            for arm in ARMS:
                payload = point[arm.field]
                for group in ("minority", "majority", "overall"):
                    value = float(payload["accuracy"][group])
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"invalid {group} accuracy for {arm.field}")
                arm_backward = int(payload["compute"]["backward_examples"])
                if backward_examples is None:
                    backward_examples = arm_backward
                elif arm_backward != backward_examples:
                    raise ValueError("MoE arms used unequal backward examples")
                arm_router = int(payload["compute"]["router_training_examples"])
                if router_training_examples is None:
                    router_training_examples = arm_router
                elif arm_router != router_training_examples:
                    raise ValueError("MoE arms used unequal router-training examples")
    if backward_examples is None:
        raise ValueError("no arm compute was found")
    if router_training_examples is None:
        raise ValueError("no router compute was found")
    return (
        seeds,
        budgets,
        backward_examples,
        router_training_examples,
        int(results[0]["initial_labels"]),
    )


def _values(
    results: list[dict[str, Any]],
    budget_index: int,
    arm: str,
    group: str,
) -> list[float]:
    return [
        float(result["points"][budget_index][arm]["accuracy"][group])
        for result in results
    ]


def _paired_differences(
    results: list[dict[str, Any]],
    budget_index: int,
    arm: str,
    group: str,
) -> list[float]:
    arm_values = _values(results, budget_index, arm, group)
    baseline = _values(results, budget_index, BASELINE, group)
    return [
        value - reference for value, reference in zip(arm_values, baseline, strict=True)
    ]


def _summary_rows(
    results: list[dict[str, Any]],
    budgets: list[int],
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for budget_index, budget in enumerate(budgets):
        first_point = results[0]["points"][budget_index]
        for arm in ARMS:
            minority = _interval(_values(results, budget_index, arm.field, "minority"))
            majority = _interval(_values(results, budget_index, arm.field, "majority"))
            minority_delta = _interval(
                _paired_differences(
                    results,
                    budget_index,
                    arm.field,
                    "minority",
                )
            )
            majority_delta = _interval(
                _paired_differences(
                    results,
                    budget_index,
                    arm.field,
                    "majority",
                )
            )
            attainment = statistics.fmean(
                float(result["points"][budget_index][arm.field]["target_attained"])
                for result in results
            )
            rows.append(
                {
                    "budget": budget,
                    "total_labels": int(first_point["total_labels"]),
                    "arm": arm.field,
                    "seed_count": len(results),
                    "minority_mean": minority.mean,
                    "minority_sd": minority.standard_deviation,
                    "minority_ci_low": minority.lower,
                    "minority_ci_high": minority.upper,
                    "majority_mean": majority.mean,
                    "majority_sd": majority.standard_deviation,
                    "majority_ci_low": majority.lower,
                    "majority_ci_high": majority.upper,
                    "minority_delta_mean": minority_delta.mean,
                    "minority_delta_ci_low": minority_delta.lower,
                    "minority_delta_ci_high": minority_delta.upper,
                    "majority_delta_mean": majority_delta.mean,
                    "majority_delta_ci_low": majority_delta.lower,
                    "majority_delta_ci_high": majority_delta.upper,
                    "joint_target_attainment": attainment,
                }
            )
    return rows


def _series(
    results: list[dict[str, Any]],
    budgets: list[int],
    arm: str,
    group: str,
    *,
    paired_delta: bool,
) -> tuple[list[float], list[float], list[float], list[list[float]]]:
    intervals: list[Interval] = []
    raw: list[list[float]] = []
    for budget_index in range(len(budgets)):
        values = (
            _paired_differences(results, budget_index, arm, group)
            if paired_delta
            else _values(results, budget_index, arm, group)
        )
        raw.append(values)
        intervals.append(_interval(values))
    return (
        [interval.mean for interval in intervals],
        [interval.lower for interval in intervals],
        [interval.upper for interval in intervals],
        raw,
    )


def _plot(
    results: list[dict[str, Any]],
    budgets: list[int],
    backward_examples: int,
    router_training_examples: int,
    output_prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), sharex=True)
    minority_axis, majority_axis = axes[0]
    minority_delta_axis, majority_delta_axis = axes[1]
    absolute_axes = {
        "minority": minority_axis,
        "majority": majority_axis,
    }
    delta_axes = {
        "minority": minority_delta_axis,
        "majority": majority_delta_axis,
    }

    for group, axis in absolute_axes.items():
        for arm in ARMS:
            means, lowers, uppers, _ = _series(
                results,
                budgets,
                arm.field,
                group,
                paired_delta=False,
            )
            marker_face = arm.color if arm.marker_face is None else arm.marker_face
            axis.fill_between(
                budgets,
                lowers,
                uppers,
                color=arm.color,
                alpha=0.10 if arm.field != BASELINE else 0.07,
                linewidth=0,
            )
            axis.plot(
                budgets,
                means,
                color=arm.color,
                linestyle=arm.linestyle,
                marker=arm.marker,
                linewidth=arm.linewidth,
                markersize=5.8,
                markerfacecolor=marker_face,
                markeredgecolor=arm.color,
                markeredgewidth=1.1,
                label=arm.label,
                zorder=4 if arm.field == "routed_specialist" else 3,
            )

    jitter = {
        "ordinary_routed": -3.0,
        "specialist_mean": -1.0,
        "permuted_routed_specialist": 1.0,
        "routed_specialist": 3.0,
    }
    for group, axis in delta_axes.items():
        axis.axhline(0, color="#202020", linewidth=1.0, alpha=0.75)
        for arm in ARMS[1:]:
            means, lowers, uppers, raw = _series(
                results,
                budgets,
                arm.field,
                group,
                paired_delta=True,
            )
            marker_face = arm.color if arm.marker_face is None else arm.marker_face
            axis.fill_between(
                budgets,
                lowers,
                uppers,
                color=arm.color,
                alpha=0.11,
                linewidth=0,
            )
            axis.plot(
                budgets,
                means,
                color=arm.color,
                linestyle=arm.linestyle,
                marker=arm.marker,
                linewidth=arm.linewidth,
                markersize=5.8,
                markerfacecolor=marker_face,
                markeredgecolor=arm.color,
                markeredgewidth=1.1,
                zorder=4 if arm.field == "routed_specialist" else 3,
            )
            for budget, seed_values in zip(budgets, raw, strict=True):
                axis.scatter(
                    [budget + jitter[arm.field]] * len(seed_values),
                    seed_values,
                    s=8,
                    color=arm.color,
                    alpha=0.16,
                    linewidths=0,
                    zorder=1,
                )

    minority_axis.set_title("A  Minority accuracy")
    majority_axis.set_title("B  Majority accuracy")
    minority_delta_axis.set_title("C  Minority gain over ordinary mean")
    majority_delta_axis.set_title("D  Majority gain over ordinary mean")
    minority_axis.set_ylabel("Clean-test accuracy")
    minority_delta_axis.set_ylabel("Paired accuracy difference")
    for axis in axes[1]:
        axis.set_xlabel("New queried labels")
    for axis in axes.flat:
        axis.set_xticks(budgets)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        axis.tick_params(labelsize=9)

    minority_axis.set_ylim(0.45, 1.0)
    majority_axis.set_ylim(0.84, 1.0)
    minority_delta_axis.set_ylim(-0.10, 0.42)
    majority_delta_axis.set_ylim(-0.12, 0.10)
    for target in (0.80, 0.85, 0.90):
        minority_axis.axhline(
            target,
            color="#AAAAAA",
            linestyle=(0, (2, 3)),
            linewidth=0.8,
            zorder=0,
        )
        minority_axis.text(
            budgets[-1] + 4,
            target,
            f"{target:.2f}",
            color="#777777",
            fontsize=8,
            va="center",
        )
    majority_axis.axhline(
        0.90,
        color="#AAAAAA",
        linestyle=(0, (2, 3)),
        linewidth=0.8,
        zorder=0,
    )
    majority_axis.text(
        budgets[-1] + 4,
        0.90,
        "0.90 floor",
        color="#777777",
        fontsize=8,
        va="center",
    )
    for axis in axes.flat:
        axis.set_xlim(budgets[0] - 10, budgets[-1] + 25)

    full_at_64 = None
    baseline_at_max = None
    if 64 in budgets:
        index_64 = budgets.index(64)
        full_at_64 = _interval(
            _values(results, index_64, "routed_specialist", "minority")
        ).mean
    baseline_at_max = _interval(
        _values(results, len(budgets) - 1, BASELINE, "minority")
    ).mean
    if full_at_64 is not None and full_at_64 >= 0.85 and baseline_at_max < 0.85:
        minority_axis.annotate(
            "Observed 0.85 mean crossing:\ntop-1 MoE ≤64 labels\n"
            f"ordinary >{budgets[-1]}  ⇒  >4x (right-censored)",
            xy=(64, full_at_64),
            xytext=(104, 0.725),
            fontsize=9,
            color="#006B50",
            arrowprops={
                "arrowstyle": "->",
                "color": "#009E73",
                "linewidth": 1.1,
            },
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#F4FBF8",
                "edgecolor": "#A7D8C7",
                "alpha": 0.96,
            },
        )

    handles, labels = minority_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        fontsize=9,
        columnspacing=1.5,
        handlelength=3,
    )
    figure.suptitle(
        "Environment-routed MoE uses labeled support more efficiently",
        fontsize=16,
        fontweight="bold",
        y=1.025,
    )
    figure.text(
        0.5,
        0.015,
        (
            f"{len(results)} paired synthetic data seeds • shading: pointwise 95% "
            f"Student-t intervals • {backward_examples:,} backward examples per "
            f"MoE arm + {router_training_examples:,} router examples • one "
            "selected expert at inference\n"
            "All 4,096 unlabeled candidates are inspected at every budget; "
            "connecting lines join tested budgets only. Group identity is "
            "evaluation-only."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4A4A4A",
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.95,
        bottom=0.12,
        top=0.88,
        hspace=0.32,
        wspace=0.22,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_prefix.with_suffix(".svg")
    figure.savefig(svg_path, bbox_inches="tight")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    figure.savefig(
        output_prefix.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def _write_artifacts(
    results: list[dict[str, Any]],
    summary_rows: list[dict[str, str | int | float]],
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
            fieldnames=list(summary_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot all sample-efficiency variants against ordinary averaging."
    )
    parser.add_argument(
        "--input",
        default="-",
        help="ssilite-sample-efficiency JSON path, or '-' for stdin",
    )
    parser.add_argument(
        "--output-prefix",
        default="artifacts/sample_efficiency_variants",
        help="path stem for .svg, .png, .csv, and normalized .json outputs",
    )
    arguments = parser.parse_args(argv)

    results = _load_results(arguments.input)
    _, budgets, backward_examples, router_training_examples, _ = _validated_axes(
        results
    )
    output_prefix = Path(arguments.output_prefix).resolve()
    summary_rows = _summary_rows(results, budgets)
    _write_artifacts(results, summary_rows, output_prefix)
    _plot(
        results,
        budgets,
        backward_examples,
        router_training_examples,
        output_prefix,
    )
    print(output_prefix.with_suffix(".svg"))
    print(output_prefix.with_suffix(".png"))
    print(output_prefix.with_suffix(".csv"))
    print(output_prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
