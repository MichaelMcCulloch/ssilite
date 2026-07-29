import inspect

import pytest
import torch
from torch import nn

from ssilite.environment_moe import (
    EnvironmentMoE,
    EnvironmentMoEConfig,
    TensorizedExpertBank,
    train_environment_moe,
)


def test_sparse_dispatch_matches_dense_reference_and_skips_unselected_gradients() -> (
    None
):
    torch.manual_seed(3)
    bank = TensorizedExpertBank(
        num_experts=3,
        input_dimensions=4,
        hidden_dimensions=5,
    )
    features = torch.randn(7, 4)
    expert_ids = torch.tensor([0, 2, 0, 2, 0, 2, 0])

    dense = bank.all_logits(features)
    sparse = bank.selected_logits(features, expert_ids)

    torch.testing.assert_close(
        sparse,
        dense[expert_ids, torch.arange(features.shape[0])],
    )
    sparse.sum().backward()
    for parameter in bank.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad[1]).item() == 0


def test_environment_moe_is_one_module_with_tensorized_experts() -> None:
    model = EnvironmentMoE(
        input_dimensions=6,
        hidden_dimensions=4,
        num_experts=3,
        feature_mean=torch.zeros(6),
        feature_scale=torch.ones(6),
    )

    assert isinstance(model, nn.Module)
    assert not any(isinstance(module, nn.ModuleList) for module in model.modules())
    assert model.experts.input_weight.shape == (3, 4, 6)
    assert model.experts.output_weight.shape == (3, 4)
    assert model.router.weight.shape == (3, 6)


def _small_problem() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    majority = torch.randn(72, 3, generator=generator)
    minority = torch.randn(24, 3, generator=generator)
    majority[:, 2] -= 4
    minority[:, 2] += 4
    train_features = torch.cat((majority, minority))
    labels = torch.cat(
        (
            (majority[:, 0] > 0).float(),
            (minority[:, 1] > 0).float(),
        )
    )
    environment_ids = torch.cat(
        (
            torch.zeros(majority.shape[0], dtype=torch.long),
            torch.ones(minority.shape[0], dtype=torch.long),
        )
    )
    test_features = torch.randn(40, 3, generator=generator)
    test_features[:20, 2] -= 4
    test_features[20:, 2] += 4
    return train_features, labels, test_features, environment_ids


def _small_config(device: str = "cpu") -> EnvironmentMoEConfig:
    return EnvironmentMoEConfig(
        hidden_dimensions=6,
        training_steps=4,
        batch_size=8,
        learning_rate=0.02,
        focus_mass=0.95,
        seed=23,
        device=device,
    )


def test_training_api_excludes_hidden_evaluation_metadata() -> None:
    assert tuple(inspect.signature(train_environment_moe).parameters) == (
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
        train_environment_moe(
            train_features,
            labels,
            test_features,
            environment_ids,
            config=_small_config(),
            clean_labels=labels,  # type: ignore[call-arg]
        )


def test_paired_arms_share_router_and_use_equal_exact_compute() -> None:
    train_features, labels, test_features, environment_ids = _small_problem()
    config = _small_config()
    result = train_environment_moe(
        train_features,
        labels,
        test_features,
        environment_ids,
        config=config,
    )

    assert result.ordinary.compute == result.specialist.compute
    assert result.ordinary.compute.model_fits == 1
    assert result.ordinary.compute.optimizer_steps == config.training_steps
    assert result.ordinary.compute.expert_updates == 2 * config.training_steps
    assert (
        result.ordinary.compute.backward_examples
        == 2 * config.training_steps * config.batch_size
    )
    assert (
        result.ordinary.compute.router_training_examples
        == config.training_steps * config.batch_size
    )
    torch.testing.assert_close(
        result.ordinary.router_logits,
        result.specialist.router_logits,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result.ordinary.routed_expert_indices,
        result.specialist.routed_expert_indices,
        rtol=0,
        atol=0,
    )
    assert result.ordinary.logits.shape == (2, test_features.shape[0])
    assert result.specialist.logits.shape == (2, test_features.shape[0])
    assert result.specialist.route_counts.sum().item() == test_features.shape[0]


def test_training_is_deterministic() -> None:
    arguments = _small_problem()
    first = train_environment_moe(*arguments, config=_small_config())
    second = train_environment_moe(*arguments, config=_small_config())

    for arm_name in ("ordinary", "specialist"):
        first_arm = getattr(first, arm_name)
        second_arm = getattr(second, arm_name)
        for field in (
            "logits",
            "probabilities",
            "mean_logits",
            "mean_probabilities",
            "router_logits",
            "router_probabilities",
            "routed_expert_indices",
            "routed_logits",
            "routed_probabilities",
            "route_counts",
            "router_train_accuracy",
        ):
            torch.testing.assert_close(
                getattr(first_arm, field),
                getattr(second_arm, field),
                rtol=0,
                atol=0,
            )


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


def test_learned_router_and_specialist_rescue_rare_coherent_rule() -> None:
    train_features, labels, test_features, test_labels, environment_ids = (
        _rare_coherent_problem()
    )
    result = train_environment_moe(
        train_features,
        labels,
        test_features,
        environment_ids,
        config=EnvironmentMoEConfig(
            hidden_dimensions=10,
            training_steps=30,
            batch_size=48,
            learning_rate=0.02,
            focus_mass=0.95,
            seed=19,
            device="cpu",
        ),
    )

    ordinary_correct = (result.ordinary.mean_probabilities >= 0.5) == test_labels.bool()
    routed_correct = (
        result.specialist.routed_probabilities >= 0.5
    ) == test_labels.bool()
    majority = torch.arange(test_labels.numel()) < 120
    rare = ~majority
    assert routed_correct[rare].float().mean() > (
        ordinary_correct[rare].float().mean() + 0.2
    )
    assert routed_correct[majority].float().mean() >= (
        ordinary_correct[majority].float().mean() - 0.08
    )
    assert torch.all(result.specialist.routed_expert_indices[majority] == 0)
    assert torch.all(result.specialist.routed_expert_indices[rare] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_training_keeps_outputs_and_parameters_on_gpu() -> None:
    train_features, labels, test_features, environment_ids = _small_problem()
    result = train_environment_moe(
        train_features,
        labels,
        test_features,
        environment_ids,
        config=_small_config("cuda"),
    )

    assert result.device.type == "cuda"
    assert result.ordinary.logits.device.type == "cuda"
    assert result.specialist.routed_logits.device.type == "cuda"
    assert result.specialist.router_logits.device.type == "cuda"
