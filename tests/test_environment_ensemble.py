import inspect
from dataclasses import replace

import pytest
import torch

from ssilite.environment_ensemble import (
    EnvironmentEnsembleConfig,
    EnvironmentEnsembleEstimator,
    discover_feature_environments,
    estimate_environment_ensemble,
)


def _small_observed_problem(seed: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(72, 4, generator=generator)
    labels = (features[:, 0] - 0.4 * features[:, 1] > 0).float()
    labels[::11] = 1 - labels[::11]
    return features, labels


def _small_config(**changes) -> EnvironmentEnsembleConfig:
    return replace(
        EnvironmentEnsembleConfig(
            num_environments=2,
            num_folds=3,
            num_repeats=1,
            hidden_dimensions=8,
            training_steps=5,
            batch_size=24,
            seed=17,
            device="cpu",
        ),
        **changes,
    )


def test_public_environment_discovery_is_label_free_and_deterministic() -> None:
    parameters = inspect.signature(discover_feature_environments).parameters
    assert tuple(parameters) == (
        "features",
        "num_environments",
        "iterations",
        "seed",
        "device",
    )
    features, _ = _small_observed_problem()

    first = discover_feature_environments(
        features,
        num_environments=2,
        iterations=5,
        seed=13,
        device="cpu",
    )
    second = discover_feature_environments(
        features,
        num_environments=2,
        iterations=5,
        seed=13,
        device="cpu",
    )

    assert first.shape == (features.shape[0],)
    assert first.dtype == torch.long
    assert torch.all(torch.bincount(first, minlength=2) > 0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_oof_prediction_never_uses_the_label_it_grades() -> None:
    features, labels = _small_observed_problem()
    changed_labels = labels.clone()
    changed_labels[7] = 1 - changed_labels[7]
    config = _small_config()

    original = estimate_environment_ensemble(features, labels, config=config)
    changed = estimate_environment_ensemble(features, changed_labels, config=config)

    # Fold construction sees features only, and example 7 is absent from every
    # model that fills its OOF prediction.  Changing its training label thus
    # cannot change its class-probability prediction.
    torch.testing.assert_close(
        original.oof_probabilities[:, :, 7],
        changed.oof_probabilities[:, :, 7],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        original.fold_assignments,
        changed.fold_assignments,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        original.label_support_by_student[:, :, 7],
        changed.label_support_by_student[:, :, 7],
    )


def test_estimate_is_deterministic_and_records_equal_fit_budget() -> None:
    features, labels = _small_observed_problem()
    config = _small_config(num_repeats=2, training_steps=4)

    first = estimate_environment_ensemble(features, labels, config=config)
    second = estimate_environment_ensemble(features, labels, config=config)

    for field in (
        "environment_ids",
        "fold_assignments",
        "student_initialization_seeds",
        "oof_probabilities",
        "label_support_by_student",
        "matched_environment_support",
        "best_environment_support",
        "learnability",
        "shared_corruption",
        "round_learnability",
        "trust_history",
        "max_trust_delta_history",
        "trust_scores",
        "trust_base_weights",
        "equal_environment_base_weights",
        "base_weights",
    ):
        torch.testing.assert_close(
            getattr(first, field),
            getattr(second, field),
            rtol=0,
            atol=0,
        )
    assert first.compute == second.compute
    assert first.compute.model_fits == 2 * 3 * 3
    assert first.compute.optimizer_steps == first.compute.model_fits * 4
    assert first.compute.scoring_forward_examples == 2 * 3 * labels.numel()
    assert first.compute.backward_examples == (
        first.compute.optimizer_steps * config.batch_size
    )


def test_base_measure_is_bounded_normalized_and_environment_balanced() -> None:
    features, labels = _small_observed_problem()
    result = estimate_environment_ensemble(
        features,
        labels,
        config=_small_config(),
    )

    assert torch.all((result.learnability >= 0) & (result.learnability <= 1))
    torch.testing.assert_close(
        result.learnability + result.shared_corruption,
        torch.ones_like(result.learnability),
    )
    assert torch.all((result.base_weights >= 0) & (result.base_weights <= 1))
    torch.testing.assert_close(result.base_weights.sum(), torch.tensor(1.0))
    torch.testing.assert_close(result.trust_base_weights.sum(), torch.tensor(1.0))
    torch.testing.assert_close(
        result.equal_environment_base_weights.sum(),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        result.environment_balanced_trust,
        result.base_weights,
        rtol=0,
        atol=0,
    )

    masses = torch.stack(
        [
            result.base_weights[result.environment_ids == environment].sum()
            for environment in result.environment_ids.unique()
        ]
    )
    torch.testing.assert_close(masses, torch.full_like(masses, 1 / masses.numel()))
    equal_masses = torch.stack(
        [
            result.equal_environment_base_weights[
                result.environment_ids == environment
            ].sum()
            for environment in result.environment_ids.unique()
        ]
    )
    torch.testing.assert_close(
        equal_masses,
        torch.full_like(equal_masses, 1 / equal_masses.numel()),
    )


def test_caller_api_cannot_receive_clean_or_group_metadata() -> None:
    parameters = inspect.signature(estimate_environment_ensemble).parameters
    assert tuple(parameters) == (
        "features",
        "observed_labels",
        "config",
        "initial_trust",
        "environment_ids",
    )
    call_parameters = inspect.signature(
        EnvironmentEnsembleEstimator.__call__
    ).parameters
    assert tuple(call_parameters) == ("self", "features", "observed_labels")

    features, labels = _small_observed_problem()
    estimator = EnvironmentEnsembleEstimator(_small_config(training_steps=2))
    weights = estimator(features, labels)
    assert estimator.last_result is not None
    torch.testing.assert_close(weights, estimator.last_result.base_weights)
    with pytest.raises(TypeError, match="unexpected keyword"):
        estimate_environment_ensemble(  # type: ignore[call-arg]
            features,
            labels,
            clean_labels=labels,
        )


def test_explicit_environment_negative_control_reuses_pairing() -> None:
    features, labels = _small_observed_problem()
    config = _small_config(training_steps=2)
    inferred = estimate_environment_ensemble(features, labels, config=config)
    permutation = torch.randperm(
        labels.numel(),
        generator=torch.Generator().manual_seed(99),
    )
    permuted_ids = inferred.environment_ids[permutation]

    negative_control = estimate_environment_ensemble(
        features,
        labels,
        config=config,
        environment_ids=permuted_ids,
    )

    torch.testing.assert_close(
        negative_control.environment_ids,
        permuted_ids,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        negative_control.fold_assignments,
        inferred.fold_assignments,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        negative_control.student_initialization_seeds,
        inferred.student_initialization_seeds,
        rtol=0,
        atol=0,
    )
    assert negative_control.compute.clustering_distance_evaluations == 0
    assert negative_control.compute.model_fits == inferred.compute.model_fits


def _rare_environment_problem() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(31)
    majority_count = 160
    rare_count = 32
    majority_core = torch.randn(majority_count, 2, generator=generator)
    rare_core = torch.randn(rare_count, 2, generator=generator)
    majority_context = -5 + 0.25 * torch.randn(majority_count, 1, generator=generator)
    rare_context = 5 + 0.25 * torch.randn(rare_count, 1, generator=generator)
    features = torch.cat(
        (
            torch.cat((majority_core, majority_context), dim=1),
            torch.cat((rare_core, rare_context), dim=1),
        )
    )
    clean_labels = torch.cat(
        (
            (majority_core[:, 0] > 0).float(),
            (rare_core[:, 1] > 0).float(),
        )
    )
    rare = torch.arange(majority_count + rare_count) >= majority_count
    flipped = torch.zeros(majority_count + rare_count, dtype=torch.bool)
    flipped[torch.randperm(majority_count, generator=generator)[:18]] = True
    flipped[majority_count + torch.randperm(rare_count, generator=generator)[:6]] = True
    observed_labels = torch.where(flipped, 1 - clean_labels, clean_labels)
    return features, observed_labels, rare, flipped


def test_environment_objectives_improve_rare_vs_noise_separation() -> None:
    features, labels, rare, flipped = _rare_environment_problem()
    common = EnvironmentEnsembleConfig(
        num_environments=2,
        num_folds=3,
        num_repeats=2,
        hidden_dimensions=12,
        training_steps=25,
        batch_size=48,
        focus_mass=0.9,
        seed=9,
        device="cpu",
    )
    environment = estimate_environment_ensemble(features, labels, config=common)
    uniform = estimate_environment_ensemble(
        features,
        labels,
        config=replace(common, mode="uniform"),
    )

    rare_clean = rare & ~flipped
    majority_clean = ~rare & ~flipped
    environment_separation = (
        environment.learnability[rare_clean].mean()
        - environment.learnability[flipped].mean()
    )
    uniform_separation = (
        uniform.learnability[rare_clean].mean() - uniform.learnability[flipped].mean()
    )
    environment_minority_relative_trust = (
        environment.learnability[rare_clean].mean()
        / environment.learnability[majority_clean].mean()
    )
    uniform_minority_relative_trust = (
        uniform.learnability[rare_clean].mean()
        / uniform.learnability[majority_clean].mean()
    )
    assert environment_separation > uniform_separation + 0.1
    assert environment_minority_relative_trust > uniform_minority_relative_trust + 0.1
    assert sorted(torch.bincount(environment.environment_ids).tolist()) == [32, 160]
    torch.testing.assert_close(
        environment.student_initialization_seeds,
        uniform.student_initialization_seeds,
        rtol=0,
        atol=0,
    )
    assert (
        environment.student_initialization_seeds.unique().numel()
        == environment.student_initialization_seeds.numel()
    )
    assert not torch.equal(
        uniform.oof_probabilities[:, 0],
        uniform.oof_probabilities[:, 1],
    )
    assert environment.compute.model_fits == uniform.compute.model_fits
    assert environment.compute.optimizer_steps == uniform.compute.optimizer_steps
    assert (
        environment.compute.scoring_forward_examples
        == uniform.compute.scoring_forward_examples
    )


def test_bootstrap_rounds_apply_damped_bounded_backaction() -> None:
    features, labels = _small_observed_problem()
    initial_trust = torch.linspace(0.2, 1.0, labels.numel())
    config = _small_config(
        rounds=3,
        trust_damping=0.5,
        max_trust_delta=0.08,
        training_steps=3,
    )

    result = estimate_environment_ensemble(
        features,
        labels,
        config=config,
        initial_trust=initial_trust,
    )

    torch.testing.assert_close(result.trust_history[0], initial_trust)
    torch.testing.assert_close(result.trust_history[-1], result.trust_scores)
    torch.testing.assert_close(result.round_learnability[-1], result.learnability)
    assert result.trust_history.shape == (4, labels.numel())
    assert result.round_learnability.shape == (3, labels.numel())
    assert torch.all(result.max_trust_delta_history <= 0.08 + 1e-7)
    assert torch.all(
        (result.trust_history[1:] - result.trust_history[:-1]).abs() <= 0.08 + 1e-7
    )
    assert not torch.equal(
        result.round_learnability[0],
        result.round_learnability[-1],
    )
    assert result.compute.model_fits == 3 * 1 * 3 * 3
    assert isinstance(result.converged, bool)


def test_permanent_distrust_is_monotone_deterministic_and_opt_in() -> None:
    features, labels = _small_observed_problem()
    initial_trust = torch.linspace(0.2, 1.0, labels.numel())
    reversible_config = _small_config(
        rounds=3,
        trust_damping=0.5,
        max_trust_delta=0.08,
        training_steps=3,
    )
    assert not reversible_config.permanent_distrust
    reversible = estimate_environment_ensemble(
        features,
        labels,
        config=reversible_config,
        initial_trust=initial_trust,
    )
    reversible_deltas = reversible.trust_history[1:] - reversible.trust_history[:-1]
    assert torch.any(reversible_deltas > 0)

    permanent_config = replace(reversible_config, permanent_distrust=True)
    first = estimate_environment_ensemble(
        features,
        labels,
        config=permanent_config,
        initial_trust=initial_trust,
    )
    second = estimate_environment_ensemble(
        features,
        labels,
        config=permanent_config,
        initial_trust=initial_trust,
    )
    permanent_deltas = first.trust_history[1:] - first.trust_history[:-1]
    assert torch.all(permanent_deltas <= 0)
    assert torch.all(permanent_deltas >= -permanent_config.max_trust_delta - 1e-7)
    assert torch.all(first.trust_history >= permanent_config.trust_floor)
    assert torch.all(first.trust_history <= 1)
    torch.testing.assert_close(
        first.trust_history,
        second.trust_history,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first.round_learnability,
        second.round_learnability,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_execution_keeps_results_on_cuda() -> None:
    features, labels = _small_observed_problem()
    result = estimate_environment_ensemble(
        features,
        labels,
        config=_small_config(device="cuda", training_steps=2),
    )

    assert result.base_weights.is_cuda
    assert result.oof_probabilities.is_cuda
    torch.testing.assert_close(
        result.base_weights.sum(),
        torch.tensor(1.0, device="cuda"),
    )
