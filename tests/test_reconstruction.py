import json
from dataclasses import asdict, replace

import torch

from ssilite.environment_ensemble import EnvironmentEnsembleConfig
from ssilite.reconstruction import (
    _rare_vs_corrupt_auroc,
    run_reconstruction,
)


def test_rare_vs_corrupt_auroc_is_tie_correct() -> None:
    scores = torch.tensor([0.9, 0.5, 0.5, 0.1, 0.8])
    clean_minority = torch.tensor([True, True, False, False, False])
    corrupted = torch.tensor([False, False, True, True, False])

    assert _rare_vs_corrupt_auroc(scores, clean_minority, corrupted) == 0.875


def test_reconstruction_keeps_student_and_objective_treatments_separate(
    monkeypatch,
) -> None:
    from ssilite import reconstruction
    from ssilite.data import make_support_problem

    def tiny_problem(*, seed: int, label_noise: float, test_minority_fraction: float):
        return make_support_problem(
            train_size=72,
            reservoir_size=144,
            test_size=192,
            core_dimensions=4,
            minority_fraction=0.25,
            test_minority_fraction=test_minority_fraction,
            label_noise=label_noise,
            seed=seed,
        )

    monkeypatch.setattr(reconstruction, "make_support_problem", tiny_problem)
    config = EnvironmentEnsembleConfig(
        num_environments=2,
        num_folds=2,
        num_repeats=1,
        hidden_dimensions=6,
        training_steps=4,
        batch_size=8,
        kmeans_iterations=4,
        rounds=1,
        seed=9,
        device="cpu",
    )
    result = run_reconstruction(
        seed=2,
        acquisition_count=16,
        learner_steps=2,
        ensemble_config=replace(config),
        device="cpu",
    )

    assert set(result.ensembles) == {
        "ordinary",
        "environment",
        "permuted_environment",
    }
    assert set(result.learners) == {
        "erm",
        "ordinary_ensemble_trust",
        "environment_ensemble_trust",
        "environment_balanced",
        "environment_adversary",
        "oracle_environment_adversary",
    }
    assert set(result.mixtures) == {
        "ordinary_mean",
        "ordinary_routed",
        "specialist_mean",
        "specialist_routed",
        "permuted_ordinary_mean",
        "permuted_ordinary_routed",
        "permuted_specialist_mean",
        "permuted_specialist_routed",
    }
    assert result.mixtures["ordinary_mean"] == result.mixtures["permuted_ordinary_mean"]
    assert result.device == "cpu"
    json.dumps(asdict(result))
    assert result.ensembles["ordinary"].model_fits
    assert result.learners["erm"].backward_examples == 2 * (72 + 16)
