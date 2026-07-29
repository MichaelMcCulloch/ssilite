import inspect

import pytest
import torch

from ssilite.environment_mixture import (
    EnvironmentMixtureConfig,
    train_environment_mixture,
)


def _small_problem() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    first = torch.randn(48, 3, generator=generator)
    second = torch.randn(16, 3, generator=generator)
    first[:, 2] -= 4
    second[:, 2] += 4
    train_features = torch.cat((first, second))
    labels = torch.cat(
        (
            (first[:, 0] > 0).float(),
            (second[:, 1] > 0).float(),
        )
    )
    environment_ids = torch.cat(
        (
            torch.zeros(first.shape[0], dtype=torch.long),
            torch.ones(second.shape[0], dtype=torch.long),
        )
    )
    test_features = torch.randn(19, 3, generator=generator)
    test_features[:10, 2] -= 4
    test_features[10:, 2] += 4
    return train_features, labels, test_features, environment_ids


def _small_config() -> EnvironmentMixtureConfig:
    return EnvironmentMixtureConfig(
        hidden_dimensions=6,
        training_steps=5,
        batch_size=16,
        learning_rate=0.02,
        seed=23,
        device="cpu",
    )


def test_api_accepts_no_clean_group_or_flip_metadata() -> None:
    assert tuple(inspect.signature(train_environment_mixture).parameters) == (
        "train_features",
        "observed_labels",
        "test_features",
        "environment_ids",
        "config",
        "train_trust",
        "device",
    )
    train_features, labels, test_features, environment_ids = _small_problem()
    with pytest.raises(TypeError, match="unexpected keyword"):
        train_environment_mixture(
            train_features,
            labels,
            test_features,
            environment_ids,
            config=_small_config(),
            clean_labels=labels,  # type: ignore[call-arg]
        )


def test_arms_pair_model_seeds_and_exact_training_budgets() -> None:
    train_features, labels, test_features, environment_ids = _small_problem()
    config = _small_config()
    result = train_environment_mixture(
        train_features,
        labels,
        test_features,
        environment_ids,
        config=config,
    )

    torch.testing.assert_close(
        result.ordinary.model_seeds,
        result.specialist.model_seeds,
        rtol=0,
        atol=0,
    )
    assert result.ordinary.compute == result.specialist.compute
    assert result.ordinary.compute.model_fits == 3
    assert result.ordinary.compute.optimizer_steps == 3 * config.training_steps
    assert result.ordinary.compute.backward_examples == (
        3 * config.training_steps * config.batch_size
    )
    assert result.ordinary.compute.train_diagnostic_forward_examples == (
        3 * labels.numel()
    )
    assert result.ordinary.compute.test_forward_examples == 3 * test_features.shape[0]
    assert result.ordinary.logits.shape == (3, test_features.shape[0])
    assert result.specialist.logits.shape == (3, test_features.shape[0])
    assert not torch.equal(result.ordinary.logits, result.specialist.logits)
    for arm in (result.ordinary, result.specialist):
        assert arm.diagnostics.error_rates.shape == (3,)
        assert arm.diagnostics.prediction_correlation.shape == (3, 3)
        torch.testing.assert_close(
            arm.mean_logits.sigmoid(),
            arm.mean_probabilities,
        )


def test_training_and_routing_are_deterministic() -> None:
    train_features, labels, test_features, environment_ids = _small_problem()
    trust = torch.linspace(0.3, 1, labels.numel())
    arguments = (
        train_features,
        labels,
        test_features,
        environment_ids,
    )
    first = train_environment_mixture(
        *arguments,
        config=_small_config(),
        train_trust=trust,
    )
    second = train_environment_mixture(
        *arguments,
        config=_small_config(),
        train_trust=trust,
    )

    for name in (
        "routing_environment_ids",
        "environment_centers",
        "feature_mean",
        "feature_scale",
    ):
        torch.testing.assert_close(
            getattr(first, name),
            getattr(second, name),
            rtol=0,
            atol=0,
        )
    for arm_name in ("ordinary", "specialist"):
        first_arm = getattr(first, arm_name)
        second_arm = getattr(second, arm_name)
        for name in (
            "logits",
            "probabilities",
            "mean_logits",
            "mean_probabilities",
            "model_seeds",
        ):
            torch.testing.assert_close(
                getattr(first_arm, name),
                getattr(second_arm, name),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            first_arm.diagnostics.prediction_correlation,
            second_arm.diagnostics.prediction_correlation,
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        first.specialist.routed_probabilities,
        second.specialist.routed_probabilities,
        rtol=0,
        atol=0,
    )


def test_nearest_center_routing_has_one_id_and_prediction_per_test_example() -> None:
    train_features, labels, test_features, environment_ids = _small_problem()
    result = train_environment_mixture(
        train_features,
        labels,
        test_features,
        environment_ids,
        config=_small_config(),
    )

    assert result.routing_environment_ids.shape == (test_features.shape[0],)
    assert result.specialist.routed_student_indices.shape == (test_features.shape[0],)
    assert result.specialist.routed_logits.shape == (test_features.shape[0],)
    assert result.specialist.routed_probabilities.shape == (test_features.shape[0],)
    assert result.environment_centers.shape == (2, train_features.shape[1])
    torch.testing.assert_close(
        result.routing_environment_ids,
        result.specialist.routed_student_indices,
    )
    assert torch.all(result.routing_environment_ids[:10] == 0)
    assert torch.all(result.routing_environment_ids[10:] == 1)


def _rare_coherent_problem() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(41)

    def block(count: int, *, rare: bool) -> tuple[torch.Tensor, torch.Tensor]:
        core = torch.randn(count, 2, generator=generator)
        location = 5 if rare else -5
        context = location + 0.3 * torch.randn(count, 1, generator=generator)
        features = torch.cat((core, context), dim=1)
        labels = (core[:, 1] > 0).float() if rare else (core[:, 0] > 0).float()
        return features, labels

    majority_train, majority_labels = block(240, rare=False)
    rare_train, rare_labels = block(24, rare=True)
    majority_test, majority_test_labels = block(120, rare=False)
    rare_test, rare_test_labels = block(120, rare=True)
    train_features = torch.cat((majority_train, rare_train))
    train_labels = torch.cat((majority_labels, rare_labels))
    test_features = torch.cat((majority_test, rare_test))
    test_labels = torch.cat((majority_test_labels, rare_test_labels))
    environment_ids = torch.cat(
        (
            torch.zeros(majority_train.shape[0], dtype=torch.long),
            torch.ones(rare_train.shape[0], dtype=torch.long),
        )
    )
    return train_features, train_labels, test_features, test_labels, environment_ids


def test_routed_specialists_rescue_rare_coherent_environment() -> None:
    train_features, labels, test_features, test_labels, environment_ids = (
        _rare_coherent_problem()
    )
    result = train_environment_mixture(
        train_features,
        labels,
        test_features,
        environment_ids,
        config=EnvironmentMixtureConfig(
            hidden_dimensions=10,
            training_steps=30,
            batch_size=48,
            learning_rate=0.02,
            focus_mass=0.95,
            seed=19,
            device="cpu",
        ),
    )

    ordinary_correct = (result.ordinary.mean_probabilities >= 0.5) == test_labels.to(
        dtype=torch.bool
    )
    routed_correct = (result.specialist.routed_probabilities >= 0.5) == test_labels.to(
        dtype=torch.bool
    )
    majority = torch.arange(test_labels.numel()) < 120
    rare = ~majority
    ordinary_rare_accuracy = ordinary_correct[rare].float().mean()
    routed_rare_accuracy = routed_correct[rare].float().mean()
    ordinary_majority_accuracy = ordinary_correct[majority].float().mean()
    routed_majority_accuracy = routed_correct[majority].float().mean()

    assert routed_rare_accuracy > ordinary_rare_accuracy + 0.2
    assert routed_majority_accuracy >= ordinary_majority_accuracy - 0.08
    assert torch.all(result.routing_environment_ids[majority] == 0)
    assert torch.all(result.routing_environment_ids[rare] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_execution_keeps_outputs_on_cuda() -> None:
    train_features, labels, test_features, environment_ids = _small_problem()
    result = train_environment_mixture(
        train_features,
        labels,
        test_features,
        environment_ids,
        config=_small_config(),
        device="cuda",
    )

    assert result.ordinary.mean_probabilities.is_cuda
    assert result.specialist.routed_probabilities.is_cuda
    assert result.routing_environment_ids.is_cuda
    assert result.environment_centers.is_cuda
