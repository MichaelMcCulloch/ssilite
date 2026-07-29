import json
from dataclasses import asdict

import torch

import ssilite.uncertainty_spawning_experiment as spawning_experiment
from ssilite.uncertainty_spawning import SpawningMoEConfig
from ssilite.uncertainty_spawning_experiment import (
    ARM_NAMES,
    UncertaintyBenchmarkConfig,
    main,
    make_uncertainty_spawning_problem,
    run_uncertainty_spawning_benchmark,
)


def _tiny_benchmark_config() -> UncertaintyBenchmarkConfig:
    return UncertaintyBenchmarkConfig(
        budgets=(192,),
        test_size=180,
        core_dimensions=3,
        rare_fraction=0.2,
        stochastic_fraction=0.2,
        context_separation=4,
        context_noise=0.25,
        common_target=0.5,
        rare_target=0.5,
        oracle_training_steps=2,
        device="cpu",
    )


def _tiny_spawning_config() -> SpawningMoEConfig:
    return SpawningMoEConfig(
        max_experts=3,
        hidden_dimensions=5,
        batch_size=16,
        learning_rate=0.03,
        warmup_count=16,
        calibration_capacity=64,
        surprise_tail_probability=0.2,
        raw_loss_threshold=0.8,
        proposal_interval=32,
        proposal_clusters=3,
        proposal_min_support=6,
        proposal_buffer_capacity=96,
        proposal_validation_fraction=0.4,
        kmeans_iterations=4,
        cooldown_examples=16,
        challenger_steps=3,
        bootstrap_samples=8,
        router_steps=3,
        replay_capacity=32,
        seed=71,
        device="cpu",
    )


def test_problem_contains_distinct_evaluation_only_strata() -> None:
    problem = make_uncertainty_spawning_problem(
        train_size=300,
        test_size=300,
        core_dimensions=3,
        rare_fraction=0.2,
        stochastic_fraction=0.2,
        seed=5,
    )
    assert torch.bincount(problem.train.stratum, minlength=3).min() > 0
    centers = torch.stack(
        [
            problem.train.features[problem.train.stratum == stratum, 3:].mean(0)
            for stratum in range(3)
        ]
    )
    assert torch.cdist(centers, centers).fill_diagonal_(100).min() > 3
    assert torch.equal(problem.train.labels, problem.train.clean_labels)


def test_small_benchmark_runs_every_causal_arm_with_strict_json() -> None:
    result = run_uncertainty_spawning_benchmark(
        seed=3,
        config=_tiny_benchmark_config(),
        spawning_config=_tiny_spawning_config(),
    )

    assert result.device == "cpu"
    assert len(result.points) == 1
    point = result.points[0]
    assert tuple(point.arms) == ARM_NAMES
    assert (
        sum((point.common_examples, point.rare_examples, point.stochastic_examples))
        == point.total_labels
    )
    for arm in point.arms.values():
        assert sum(arm.route_counts) == _tiny_benchmark_config().test_size
        assert arm.compute
    encoded = json.dumps(asdict(result), allow_nan=False, sort_keys=True)
    assert json.loads(encoded)["seed"] == 3


def test_small_benchmark_is_deterministic() -> None:
    arguments = {
        "seed": 9,
        "config": _tiny_benchmark_config(),
        "spawning_config": _tiny_spawning_config(),
    }
    first = run_uncertainty_spawning_benchmark(**arguments)
    second = run_uncertainty_spawning_benchmark(**arguments)
    assert asdict(first) == asdict(second)


def test_driver_cli_prints_strict_json(monkeypatch, capsys) -> None:
    result = run_uncertainty_spawning_benchmark(
        seed=0,
        config=_tiny_benchmark_config(),
        spawning_config=_tiny_spawning_config(),
    )
    monkeypatch.setattr(
        spawning_experiment,
        "run_uncertainty_spawning_benchmark",
        lambda **_kwargs: result,
    )

    main(
        [
            "--seeds",
            "2",
            "3",
            "--budgets",
            "192",
            "--test-size",
            "180",
            "--device",
            "cpu",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    assert set(payload[0]["points"][0]["arms"]) == set(ARM_NAMES)
