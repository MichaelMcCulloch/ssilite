import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "plot_uncertainty_spawning.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "plot_uncertainty_spawning",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
plotting = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = plotting
_SPEC.loader.exec_module(plotting)


def _arm(seed: int, budget: int, arm_index: int) -> dict:
    return {
        "accuracy": {
            "overall": 0.7 + 0.01 * seed,
            "common": 0.8 + 0.01 * arm_index,
            "rare_rule": 0.6 + 0.01 * budget / 100,
            "stochastic_pocket": 0.5,
        },
        "active_experts": 1 + (arm_index == 4),
        "birth_count": int(arm_index in {1, 3, 4}),
        "rare_rule_births": int(arm_index in {3, 4}),
        "stochastic_false_births": int(arm_index == 1),
        "surprise_rate": 0.1,
        "proposal_count": 2,
        "stochastic_rejections": int(arm_index == 4),
        "expected_uncertainties": [0.2],
        "rare_evidence_mean": 1.5,
        "stochastic_evidence_mean": -0.5,
        "birth_examples": [],
        "birth_evidence": [],
        "route_counts": [40, 0, 0],
        "common_route_counts": [24, 0, 0],
        "rare_route_counts": [8, 0, 0],
        "stochastic_route_counts": [8, 0, 0],
        "compute": {
            "task_forward_examples": budget,
            "task_backward_examples": budget,
            "controller_scoring_forward_examples": budget,
            "sparse_inference_examples": 40,
        },
    }


def _results() -> list[dict]:
    results = []
    for seed in (0, 1):
        points = []
        for budget in (100, 200):
            points.append(
                {
                    "total_labels": budget,
                    "arms": {
                        style.field: _arm(seed, budget, arm_index)
                        for arm_index, style in enumerate(plotting.ARMS)
                    },
                }
            )
        results.append({"seed": seed, "device": "cpu", "points": points})
    return results


def test_plotter_validates_axes_and_writes_long_form_summary() -> None:
    results = _results()
    seeds, budgets = plotting._validated_axes(results)
    rows = plotting._summary_rows(results)

    assert seeds == [0, 1]
    assert budgets == [100, 200]
    assert len(rows) == 2 * 2 * len(plotting.ARMS)
    assert rows[0]["forward_work"] == 240
    assert rows[0]["backward_work"] == 100
    means, half_widths = plotting._series(
        results,
        arm="joint",
        field="rare_rule",
    )
    assert means == [0.61, 0.62]
    assert half_widths == [0.0, 0.0]


def test_plotter_cli_writes_json_and_csv(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(_results()), encoding="utf-8")
    output_prefix = tmp_path / "figure"
    plotted = []
    monkeypatch.setattr(
        plotting,
        "_plot",
        lambda results, budgets, prefix: plotted.append(
            (len(results), budgets, prefix)
        ),
    )

    plotting.main(
        [
            "--input",
            str(input_path),
            "--output-prefix",
            str(output_prefix),
        ]
    )

    assert plotted == [(2, [100, 200], output_prefix.resolve())]
    assert output_prefix.with_suffix(".json").exists()
    assert output_prefix.with_suffix(".csv").exists()
    assert len(capsys.readouterr().out.splitlines()) == 4
