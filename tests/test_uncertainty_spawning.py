import inspect
import json
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from ssilite.uncertainty_spawning import (
    ChallengerDecision,
    EmpiricalCalibration,
    Proposal,
    SpawningMoE,
    SpawningMoEConfig,
    UnexpectedUncertaintyController,
    evaluate_birth_proposal,
    train_spawning_moe,
)


def _model(*, max_experts: int = 3, seed: int = 7) -> SpawningMoE:
    model = SpawningMoE(
        input_dimensions=3,
        hidden_dimensions=8,
        max_experts=max_experts,
        feature_mean=torch.zeros(3),
        feature_scale=torch.ones(3),
    )
    model.initialize(torch.Generator().manual_seed(seed))
    return model


def test_dormant_experts_are_zero_gradient_and_excluded_from_routes() -> None:
    model = _model()
    features = torch.randn(17, 3, generator=torch.Generator().manual_seed(3))
    logits, routes = model(features)

    assert torch.all(routes == 0)
    assert model.active_router_logits(features).shape == (17, 1)
    for parameter in (
        model.experts.input_weight,
        model.experts.input_bias,
        model.experts.output_weight,
        model.experts.output_bias,
        model.router.weight,
        model.router.bias,
    ):
        assert torch.count_nonzero(parameter.detach()[1:]).item() == 0

    logits.sum().backward()
    for parameter in model.experts.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad[1:]).item() == 0


def test_plain_experts_omit_shared_and_rms_paths() -> None:
    model = SpawningMoE(
        input_dimensions=3,
        hidden_dimensions=8,
        expert_architecture="plain",
        max_experts=2,
        feature_mean=torch.zeros(3),
        feature_scale=torch.ones(3),
    )
    model.initialize(torch.Generator().manual_seed(7))

    assert model.shared_output is None
    assert model.routed_norm is None
    assert model.latent_dimensions == model.hidden_dimensions


def test_activation_is_contiguous_deterministic_and_capacity_bounded() -> None:
    first = _model(max_experts=2)
    second = _model(max_experts=2)
    parameters = first.parent_parameters(0)
    router_weight = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    router_bias = torch.zeros(2)

    assert (
        first.activate(
            expert_parameters=parameters,
            router_weight=router_weight,
            router_bias=router_bias,
        )
        == 1
    )
    second.activate(
        expert_parameters=parameters,
        router_weight=router_weight,
        router_bias=router_bias,
    )
    torch.testing.assert_close(first.active_expert_mask, torch.tensor([True, True]))
    torch.testing.assert_close(first.router.weight, second.router.weight)
    with pytest.raises(RuntimeError, match="capacity"):
        first.activate(
            expert_parameters=parameters,
            router_weight=router_weight,
            router_bias=router_bias,
        )


def test_empirical_tail_calibration_uses_full_history_and_warmup() -> None:
    calibration = EmpiricalCalibration(
        max_experts=2,
        capacity=3,
        warmup_count=2,
    )
    calibration.add(0, 0.1)
    assert calibration.tail_probability(0, 100.0) == 1.0
    calibration.add(0, 0.2)
    assert calibration.tail_probability(0, 0.15) == pytest.approx(2 / 3)
    assert calibration.tail_probability(0, 10.0) == pytest.approx(1 / 3)
    calibration.add(0, 0.3)
    calibration.add(0, 0.4)
    assert calibration.count(0) == 3
    assert calibration.tail_probability(0, 0.35) == pytest.approx(2 / 4)
    assert calibration.expected_uncertainty(0) == pytest.approx(1 / 5)


def _controller_config() -> SpawningMoEConfig:
    return SpawningMoEConfig(
        max_experts=2,
        warmup_count=2,
        calibration_capacity=16,
        surprise_tail_probability=0.34,
        proposal_interval=4,
        proposal_clusters=1,
        proposal_min_support=2,
        proposal_buffer_capacity=8,
        proposal_validation_fraction=0.5,
        cooldown_examples=2,
        challenger_steps=1,
        bootstrap_samples=4,
        router_steps=1,
        replay_capacity=4,
        device="cpu",
    )


def test_unresolved_examples_are_excluded_until_rejection_and_json_is_stable() -> None:
    controllers = [
        UnexpectedUncertaintyController(_controller_config(), "joint") for _ in range(2)
    ]
    proposals = []
    for controller in controllers:
        assert not controller.observe_batch(
            example_ids=torch.tensor([0, 1]),
            features=torch.tensor([[0.0, 0.0], [0.1, 0.1]]),
            labels=torch.tensor([0.0, 1.0]),
            active_losses=torch.tensor([[0.1, 0.2]]),
        )
        proposal = controller.observe_batch(
            example_ids=torch.tensor([2, 3]),
            features=torch.tensor([[4.0, 0.0], [4.1, 0.1]]),
            labels=torch.tensor([0.0, 1.0]),
            active_losses=torch.tensor([[10.0, 10.0]]),
        )[0]
        proposals.append(proposal)
        assert controller.calibration.count(0) == 2
        assert controller.unresolved_count == 0
        controller.resolve(
            proposal,
            ChallengerDecision(
                accepted=False,
                reason="nonpositive_confidence_bound",
                evidence=None,
                expert_parameters=None,
                router_weight=None,
                router_bias=None,
            ),
        )
        assert controller.calibration.count(0) == 4
    assert controllers[0].diagnostics_json() == controllers[1].diagnostics_json()
    assert (
        json.loads(controllers[0].diagnostics_json())["rejections"][0]["reason"]
        == "nonpositive_confidence_bound"
    )
    torch.testing.assert_close(proposals[0].fit_indices, proposals[1].fit_indices)


def _trained_incumbent() -> tuple[SpawningMoE, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    features = torch.randn(160, 3, generator=generator)
    features[:, 2] = -4 + 0.2 * features[:, 2]
    labels = (features[:, 0] >= 0).float()
    model = _model(max_experts=3, seed=11)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    routes = torch.zeros(features.shape[0], dtype=torch.long)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(
            model.selected_expert_logits(features, routes),
            labels,
        )
        loss.backward()
        optimizer.step()
    return model, features[:64], labels[:64]


def _proposal(*, random_labels: bool, proposal_id: int) -> Proposal:
    generator = torch.Generator().manual_seed(101 + proposal_id)
    features = torch.randn(100, 3, generator=generator)
    features[:, 2] = 4 + 0.2 * features[:, 2]
    if random_labels:
        labels = torch.randint(2, (100,), generator=generator).float()
    else:
        labels = (features[:, 1] >= 0).float()
    permutation = torch.randperm(100, generator=generator)
    return Proposal(
        proposal_id=proposal_id,
        created_at=100,
        example_ids=torch.arange(100),
        features=features,
        labels=labels,
        losses=torch.ones(1, 100),
        fit_indices=permutation[:60],
        validation_indices=permutation[60:],
    )


def _evidence_config() -> SpawningMoEConfig:
    return SpawningMoEConfig(
        max_experts=3,
        hidden_dimensions=8,
        challenger_steps=80,
        challenger_learning_rate=0.03,
        challenger_anchor_weight=0.25,
        bootstrap_samples=128,
        practical_margin=0.02,
        router_steps=80,
        router_learning_rate=0.04,
        router_min_proposal_accuracy=0.8,
        router_min_anchor_accuracy=0.5,
        collateral_tolerance=0.08,
        seed=29,
        device="cpu",
    )


def test_random_label_pocket_fails_but_coherent_rule_passes_held_out_evidence() -> None:
    model, common_anchors, common_anchor_labels = _trained_incumbent()
    random_anchors = _proposal(random_labels=True, proposal_id=10)
    coherent_anchors = _proposal(random_labels=False, proposal_id=11)
    random_replay = torch.cat((common_anchors, random_anchors.features[:48]))
    random_replay_labels = torch.cat((common_anchor_labels, random_anchors.labels[:48]))
    coherent_replay = torch.cat((common_anchors, coherent_anchors.features[:48]))
    coherent_replay_labels = torch.cat(
        (common_anchor_labels, coherent_anchors.labels[:48])
    )
    config = _evidence_config()

    random_decision = evaluate_birth_proposal(
        model,
        _proposal(random_labels=True, proposal_id=0),
        random_replay,
        random_replay_labels,
        torch.zeros(random_replay.shape[0], dtype=torch.long),
        mode="joint",
        config=config,
    )
    coherent_decision = evaluate_birth_proposal(
        model,
        _proposal(random_labels=False, proposal_id=1),
        coherent_replay,
        coherent_replay_labels,
        torch.zeros(coherent_replay.shape[0], dtype=torch.long),
        mode="joint",
        config=config,
    )
    prototype_decision = evaluate_birth_proposal(
        model,
        _proposal(random_labels=False, proposal_id=2),
        coherent_replay,
        coherent_replay_labels,
        torch.zeros(coherent_replay.shape[0], dtype=torch.long),
        mode="joint",
        config=replace(config, routing_strategy="prototype"),
    )

    assert not random_decision.accepted
    assert random_decision.evidence is not None
    assert random_decision.evidence.lower_confidence_bound <= 0
    assert coherent_decision.accepted
    assert coherent_decision.reason == "accepted_held_out_rule"
    assert coherent_decision.evidence is not None
    assert coherent_decision.evidence.lower_confidence_bound > 0
    assert coherent_decision.evidence.unexpected_uncertainty > 0.9
    assert coherent_decision.evidence.context_switch_log_margin > 0
    assert prototype_decision.accepted
    assert prototype_decision.evidence is not None
    assert prototype_decision.evidence.anchor_route_accuracy > 0.85


def test_training_api_has_no_hidden_metadata_and_counts_exact_base_work() -> None:
    assert tuple(inspect.signature(train_spawning_moe).parameters) == (
        "train_features",
        "observed_labels",
        "test_features",
        "mode",
        "config",
        "device",
    )
    generator = torch.Generator().manual_seed(5)
    train_features = torch.randn(33, 3, generator=generator)
    labels = (train_features[:, 0] > 0).float()
    test_features = torch.randn(11, 3, generator=generator)
    config = replace(
        _controller_config(),
        batch_size=16,
        proposal_interval=100,
    )
    result = train_spawning_moe(
        train_features,
        labels,
        test_features,
        mode="single",
        config=config,
    )

    assert result.compute.task_forward_examples == 33
    assert result.compute.task_backward_examples == 33
    assert result.compute.controller_scoring_forward_examples == 33
    assert result.compute.candidate_fit_forward_examples == 0
    assert result.compute.router_training_forward_examples == 0
    assert result.compute.sparse_inference_examples == 11
    assert result.predictions.route_counts.tolist() == [11, 0]


def test_training_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(48, 3, generator=generator)
    labels = (features[:, 0] > 0).float()
    test = torch.randn(13, 3, generator=generator)
    config = replace(
        _controller_config(),
        batch_size=16,
        proposal_interval=100,
    )
    first = train_spawning_moe(features, labels, test, config=config)
    second = train_spawning_moe(features, labels, test, config=config)

    torch.testing.assert_close(
        first.predictions.routed_logits,
        second.predictions.routed_logits,
        rtol=0,
        atol=0,
    )
    assert first.compute == second.compute
    assert first.diagnostics_json == second.diagnostics_json


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_execution_keeps_predictions_on_cuda() -> None:
    generator = torch.Generator().manual_seed(23)
    features = torch.randn(32, 3, generator=generator)
    labels = (features[:, 0] > 0).float()
    result = train_spawning_moe(
        features,
        labels,
        features[:8],
        mode="single",
        config=replace(
            _controller_config(),
            proposal_interval=100,
            device="cuda",
        ),
    )
    assert result.device.type == "cuda"
    assert result.predictions.routed_logits.device.type == "cuda"
