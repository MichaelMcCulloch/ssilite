import json
from dataclasses import asdict

from ssilite import sample_efficiency
from ssilite.data import make_support_problem
from ssilite.environment_mixture import EnvironmentMixtureConfig
from ssilite.sample_efficiency import (
    SampleEfficiencyConfig,
    main,
    run_sample_efficiency,
)


def _tiny_problem(*, seed: int, label_noise: float, test_minority_fraction: float):
    return make_support_problem(
        train_size=36,
        reservoir_size=60,
        test_size=80,
        core_dimensions=4,
        minority_fraction=0.25,
        test_minority_fraction=test_minority_fraction,
        label_noise=label_noise,
        context_separation=3,
        seed=seed,
    )


def _tiny_config(**changes) -> SampleEfficiencyConfig:
    values = {
        "budgets": (0, 4, 8),
        "minority_target": 0,
        "majority_floor": 0,
        "label_noise": 0,
        "acquisition_clusters": 2,
        "num_environments": 2,
        "kmeans_iterations": 4,
        "device": "cpu",
    }
    values.update(changes)
    return SampleEfficiencyConfig(**values)


def _tiny_mixture_config() -> EnvironmentMixtureConfig:
    return EnvironmentMixtureConfig(
        hidden_dimensions=5,
        training_steps=2,
        batch_size=6,
        learning_rate=0.02,
        seed=71,
        device="cpu",
    )


def test_benchmark_uses_one_acquisition_order_and_nested_prefixes(
    monkeypatch,
) -> None:
    make_calls = 0
    acquisition_calls = 0
    mixture_calls = 0
    acquisition_result = None
    original_acquire = sample_efficiency.acquire_for_cluster_coverage
    original_mixture = sample_efficiency.train_environment_mixture

    def counted_problem(**kwargs):
        nonlocal make_calls
        make_calls += 1
        return _tiny_problem(**kwargs)

    def counted_acquisition(*args, **kwargs):
        nonlocal acquisition_calls, acquisition_result
        acquisition_calls += 1
        acquisition_result = original_acquire(*args, **kwargs)
        return acquisition_result

    def checked_mixture(*args, **kwargs):
        nonlocal mixture_calls
        mixture_calls += 1
        assert "train_trust" not in kwargs
        return original_mixture(*args, **kwargs)

    monkeypatch.setattr(sample_efficiency, "make_support_problem", counted_problem)
    monkeypatch.setattr(
        sample_efficiency,
        "acquire_for_cluster_coverage",
        counted_acquisition,
    )
    monkeypatch.setattr(
        sample_efficiency,
        "train_environment_mixture",
        checked_mixture,
    )
    result = run_sample_efficiency(
        seed=3,
        config=_tiny_config(),
        mixture_config=_tiny_mixture_config(),
    )

    assert make_calls == 1
    assert acquisition_calls == 1
    assert mixture_calls == 6
    assert acquisition_result is not None
    problem = _tiny_problem(seed=3, label_noise=0, test_minority_fraction=0.5)
    for point, budget in zip(result.points, (0, 4, 8), strict=True):
        expected_rare = int(
            problem.reservoir.minority[acquisition_result.indices[:budget]].sum().item()
        )
        assert point.new_labels == budget
        assert point.total_labels == 36 + budget
        assert point.acquired_rare_examples == expected_rare
        assert point.total_rare_examples == result.initial_rare_examples + expected_rare
        assert sum(point.environment_cluster_sizes) == point.total_labels
        assert (
            point.permuted_environment_cluster_sizes == point.environment_cluster_sizes
        )
        assert point.model_seeds == result.points[0].model_seeds
        assert point.permuted_model_seeds == point.model_seeds


def test_paired_arms_report_equal_exact_compute_and_default_budget(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sample_efficiency,
        "make_support_problem",
        _tiny_problem,
    )
    mixture_config = _tiny_mixture_config()
    result = run_sample_efficiency(
        seed=5,
        config=_tiny_config(budgets=(0,)),
        mixture_config=mixture_config,
    )
    point = result.points[0]

    arms = (
        point.ordinary_mean,
        point.ordinary_routed,
        point.specialist_mean,
        point.routed_specialist,
        point.permuted_routed_specialist,
    )
    assert all(arm.compute == arms[0].compute for arm in arms)
    assert point.ordinary_mean.compute.model_fits == 3
    assert point.ordinary_mean.compute.optimizer_steps == 6
    assert point.ordinary_mean.compute.backward_examples == 3 * 2 * 6
    assert point.ordinary_mean.compute.train_diagnostic_forward_examples == 3 * 36
    assert point.ordinary_mean.compute.test_forward_examples == 3 * 80
    default_students = SampleEfficiencyConfig().num_environments + int(
        EnvironmentMixtureConfig().include_balanced_student
    )
    assert (
        default_students
        * EnvironmentMixtureConfig().training_steps
        * EnvironmentMixtureConfig().batch_size
        == 20_480
    )


def test_results_are_deterministic_targeted_and_strict_json(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sample_efficiency,
        "make_support_problem",
        _tiny_problem,
    )
    config = _tiny_config()
    mixture_config = _tiny_mixture_config()

    first = run_sample_efficiency(
        seed=7,
        config=config,
        mixture_config=mixture_config,
    )
    second = run_sample_efficiency(
        seed=7,
        config=config,
        mixture_config=mixture_config,
    )

    assert asdict(first) == asdict(second)
    assert all(point.ordinary_mean.target_attained for point in first.points)
    assert all(point.ordinary_routed.target_attained for point in first.points)
    assert all(point.specialist_mean.target_attained for point in first.points)
    assert all(point.routed_specialist.target_attained for point in first.points)
    assert all(
        point.permuted_routed_specialist.target_attained for point in first.points
    )
    assert first.ordinary_target_crossing.status == "left_censored"
    assert first.ordinary_target_crossing.upper_new_labels == 0
    assert first.ordinary_routed_target_crossing.status == "left_censored"
    assert first.specialist_mean_target_crossing.status == "left_censored"
    assert first.routed_specialist_target_crossing.status == "left_censored"
    assert first.permuted_routed_specialist_target_crossing.status == "left_censored"
    json.dumps(asdict(first), allow_nan=False)


def test_cli_main_prints_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sample_efficiency,
        "make_support_problem",
        _tiny_problem,
    )
    result = run_sample_efficiency(
        seed=0,
        config=_tiny_config(budgets=(0,)),
        mixture_config=_tiny_mixture_config(),
    )
    monkeypatch.setattr(
        sample_efficiency,
        "run_sample_efficiency",
        lambda **_kwargs: result,
    )

    main(
        [
            "--seeds",
            "2",
            "3",
            "--budgets",
            "0",
            "--num-environments",
            "2",
            "--acquisition-clusters",
            "2",
            "--device",
            "cpu",
        ]
    )

    parsed = json.loads(capsys.readouterr().out)
    assert len(parsed) == 2
    point = parsed[0]["points"][0]
    assert point["new_labels"] == 0
    assert {
        "ordinary_mean",
        "ordinary_routed",
        "specialist_mean",
        "routed_specialist",
        "permuted_routed_specialist",
    } <= point.keys()
    assert (
        point["permuted_environment_cluster_sizes"]
        == point["environment_cluster_sizes"]
    )
    assert point["permuted_model_seeds"] == point["model_seeds"]
