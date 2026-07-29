import copy
from dataclasses import asdict

import torch

from ssilite import experiment
from ssilite.controller import ControllerConfig
from ssilite.data import DatasetSplit, make_support_problem
from ssilite.model import MechanismMLP


def _tiny_problem(*, seed: int):
    return make_support_problem(
        train_size=48,
        reservoir_size=96,
        test_size=128,
        core_dimensions=4,
        minority_fraction=0.25,
        seed=seed,
    )


def test_raw_loss_training_does_not_depend_on_clean_labels() -> None:
    split = _tiny_problem(seed=7).train
    changed_metadata = DatasetSplit(
        features=split.features,
        labels=split.labels,
        clean_labels=1 - split.clean_labels,
        minority=split.minority,
        flipped=split.flipped,
    )

    torch.manual_seed(11)
    template = MechanismMLP(split.features.shape[1], hidden_dimensions=8)
    initial_state = copy.deepcopy(template.state_dict())
    models = [
        MechanismMLP(split.features.shape[1], hidden_dimensions=8),
        MechanismMLP(split.features.shape[1], hidden_dimensions=8),
    ]
    for model in models:
        model.load_state_dict(initial_state)

    config = ControllerConfig(
        dual_step=1.0,
        precision_levels=(32,),
        precision_costs=(1.0,),
        mean_precision_budget=1.01,
        allocation_rounds=1,
    )
    for model, current_split in zip(
        models,
        (split, changed_metadata),
        strict=True,
    ):
        experiment._train_joint(
            model,
            current_split,
            steps=3,
            batch_size=8,
            learning_rate=0.01,
            seed=19,
            config=config,
            reference_mode="raw_loss",
        )

    for left, right in zip(
        models[0].parameters(),
        models[1].parameters(),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_default_result_keeps_the_original_top_level_and_arm_structure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(experiment, "make_support_problem", _tiny_problem)

    result = experiment.run_experiment(
        seed=3,
        steps=1,
        batch_size=4,
        acquisition_count=8,
        device="cpu",
    )

    assert set(asdict(result)) == {
        "seed",
        "initial_support_size",
        "initial_minority_count",
        "acquired_count",
        "acquired_minority_count",
        "acquired_flipped_count",
        "arms",
        "caveat",
    }
    assert tuple(result.arms) == (
        "erm_fixed",
        "erm_acquired",
        "joint_fixed",
        "joint_acquired",
    )
    assert result.caveat == (
        "Per-example precision is statistically emulated, not a wall-clock "
        "claim. Reducible scores use generator-clean labels as an oracle "
        "control so this experiment isolates support acquisition."
    )
    assert result.arms["joint_fixed"].scoring_forward_examples == 48
    assert result.arms["joint_fixed"].backward_examples == 4
    assert result.arms["erm_fixed"].scoring_forward_examples == 0
    assert result.arms["erm_fixed"].backward_examples == 4


def test_full_precision_controls_use_exact_ordinary_backprop(monkeypatch) -> None:
    def unexpected_quantized_estimator(*args, **kwargs):
        raise AssertionError("full-precision controls must not use quantization")

    monkeypatch.setattr(
        experiment,
        "apply_batched_binary_gradients",
        unexpected_quantized_estimator,
    )

    def unexpected_precision_allocation(*args, **kwargs):
        raise AssertionError("q-only and q+p controls must stop before precision")

    monkeypatch.setattr(
        experiment.JointController,
        "allocate",
        unexpected_precision_allocation,
    )
    split = _tiny_problem(seed=2).train
    torch.manual_seed(13)
    initial_state = copy.deepcopy(
        MechanismMLP(split.features.shape[1], hidden_dimensions=8).state_dict()
    )

    for allocation_mode in ("q_only", "q_p"):
        model = MechanismMLP(split.features.shape[1], hidden_dimensions=8)
        model.load_state_dict(initial_state)
        diagnostics = experiment._train_joint(
            model,
            split,
            steps=1,
            batch_size=4,
            learning_rate=0.01,
            seed=23,
            config=ControllerConfig(),
            reference_mode="raw_loss",
            allocation_mode=allocation_mode,
        )
        assert diagnostics.mean_quantization_mse == 0


def test_cuda_smoke_executes_models_and_estimators_on_selected_gpu(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return
    monkeypatch.setattr(experiment, "make_support_problem", _tiny_problem)

    result = experiment.run_experiment(
        seed=17,
        steps=1,
        batch_size=4,
        acquisition_count=8,
        reference_mode="raw_loss",
        device="cuda",
    )

    assert "Executed on cuda:" in result.caveat
    assert torch.cuda.get_device_name(0) in result.caveat
    assert result.arms["joint_fixed"].backward_examples == 4


def test_causal_allocation_arms_are_paired_and_reproducible(monkeypatch) -> None:
    monkeypatch.setattr(experiment, "make_support_problem", _tiny_problem)
    estimator_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def observed_base_weights(
        features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        estimator_calls.append((features, labels))
        return torch.ones_like(labels)

    arguments = {
        "seed": 5,
        "steps": 2,
        "batch_size": 4,
        "acquisition_count": 8,
        "reference_mode": "raw_loss",
        "base_weights": observed_base_weights,
        "causal_allocation_arms": True,
        "device": "cpu",
    }
    first = experiment.run_experiment(**arguments)
    second = experiment.run_experiment(**arguments)

    assert asdict(first) == asdict(second)
    assert tuple(first.arms) == (
        "erm_fixed",
        "erm_acquired",
        "joint_q_only_fixed",
        "joint_q_only_acquired",
        "joint_qp_fixed",
        "joint_qp_acquired",
        "joint_fixed",
        "joint_acquired",
    )
    # Resolve the estimator once per support and reuse its base measure for all
    # three paired allocation arms.
    assert len(estimator_calls) == 4
    assert all(call[0].shape[0] == call[1].numel() for call in estimator_calls)

    for support_name in ("fixed", "acquired"):
        q_only = first.arms[f"joint_q_only_{support_name}"]
        q_p = first.arms[f"joint_qp_{support_name}"]
        q_p_b = first.arms[f"joint_{support_name}"]
        assert q_only.mean_quantization_mse == 0
        assert q_p.mean_quantization_mse == 0
        assert q_only.mean_precision_cost == 8
        assert q_p.mean_precision_cost == 8
        assert q_p_b.mean_precision_cost <= 2
        assert q_only.backward_examples == q_p.backward_examples
        assert q_p.backward_examples == q_p_b.backward_examples
        assert q_only.scoring_forward_examples == q_p.scoring_forward_examples
        assert q_p.scoring_forward_examples == q_p_b.scoring_forward_examples
