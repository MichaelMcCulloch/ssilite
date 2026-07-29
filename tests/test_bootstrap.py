import inspect

import torch

from ssilite.bootstrap import (
    BootstrapConfig,
    average_midrank,
    estimate_cross_fitted_trust,
    make_repeated_folds,
)


def _fast_config(**overrides: object) -> BootstrapConfig:
    settings: dict[str, object] = {
        "folds": 3,
        "repeats": 2,
        "rounds": 2,
        "training_steps": 24,
        "checkpoints": 4,
        "hidden_dimensions": 8,
        "learning_rate": 0.03,
        "ema_decay": 0.4,
        "max_delta": 0.15,
        "seed": 19,
    }
    settings.update(overrides)
    return BootstrapConfig(**settings)


def test_average_midrank_is_tie_safe_neutral_and_permutation_equivariant() -> None:
    values = torch.tensor([5.0, 1.0, 5.0, 3.0, 3.0])
    ranks = average_midrank(values)
    torch.testing.assert_close(
        ranks,
        torch.tensor([0.875, 0.0, 0.875, 0.375, 0.375]),
    )

    constant = torch.ones(7)
    torch.testing.assert_close(average_midrank(constant), torch.full((7,), 0.5))
    torch.testing.assert_close(
        average_midrank(torch.tensor([2, 2, 8])),
        torch.tensor([0.25, 0.25, 1.0]),
    )

    permutation = torch.tensor([3, 0, 4, 1, 2])
    torch.testing.assert_close(
        average_midrank(values[permutation]),
        ranks[permutation],
    )
    assert torch.all((ranks >= 0) & (ranks <= 1))


def test_repeated_folds_are_balanced_deterministic_and_cover_every_point() -> None:
    first = make_repeated_folds(17, folds=4, repeats=3, seed=7)
    second = make_repeated_folds(17, folds=4, repeats=3, seed=7)
    torch.testing.assert_close(first, second)

    assert first.shape == (3, 17)
    for assignments in first:
        counts = torch.bincount(assignments, minlength=4)
        assert int(counts.max() - counts.min()) <= 1
        for fold in range(4):
            heldout = assignments == fold
            training = ~heldout
            assert torch.all(~(heldout & training))
            assert torch.all(heldout | training)


def test_every_score_is_oof_and_heldout_prediction_does_not_see_its_label() -> None:
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(30, 3, generator=generator)
    labels = (features[:, 0] > 0).to(dtype=torch.float32)
    config = _fast_config(rounds=1)
    folds = make_repeated_folds(
        len(labels),
        folds=config.folds,
        repeats=config.repeats,
        seed=config.seed,
    )

    original = estimate_cross_fitted_trust(
        features,
        labels,
        config=config,
        fold_ids=folds,
    )
    altered_labels = labels.clone()
    altered_labels[0] = 1 - altered_labels[0]
    altered = estimate_cross_fitted_trust(
        features,
        altered_labels,
        config=config,
        fold_ids=folds,
    )

    assert torch.all(original.oof_counts == config.repeats)
    assert torch.all(original.history[0].oof_counts == config.repeats)
    # Point zero's label can change its held-out loss, but no model used to
    # predict point zero was allowed to train on that label.
    torch.testing.assert_close(
        original.history[0].oof_probabilities[:, 0],
        altered.history[0].oof_probabilities[:, 0],
        rtol=0,
        atol=0,
    )


def test_estimator_is_deterministic_bounded_and_respects_max_delta() -> None:
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(36, 4, generator=generator)
    labels = (features[:, 0] - 0.25 * features[:, 1] > 0).to(torch.float32)
    config = _fast_config(rounds=3, max_delta=0.08)

    first = estimate_cross_fitted_trust(features, labels, config=config)
    second = estimate_cross_fitted_trust(features, labels, config=config)

    torch.testing.assert_close(first.trust, second.trust, rtol=0, atol=0)
    torch.testing.assert_close(first.suspicion, second.suspicion, rtol=0, atol=0)
    torch.testing.assert_close(first.fold_ids, second.fold_ids)
    assert torch.all((first.suspicion >= 0) & (first.suspicion <= 1))
    assert config.min_trust <= float(first.trust.min())
    assert float(first.trust.max()) <= 1
    for round_result in first.history:
        assert round_result.max_trust_change <= config.max_delta + 1e-7
        assert torch.all(round_result.trust >= config.min_trust)
        assert torch.all(round_result.trust <= 1)

    permutation = torch.randperm(len(labels), generator=generator)
    permuted = estimate_cross_fitted_trust(
        features[permutation],
        labels[permutation],
        config=config,
        fold_ids=first.fold_ids[:, permutation],
    )
    torch.testing.assert_close(permuted.trust, first.trust[permutation])
    torch.testing.assert_close(permuted.suspicion, first.suspicion[permutation])


def test_constant_signals_are_neutral_and_converge_without_backaction() -> None:
    features = torch.zeros(24, 2)
    labels = torch.ones(24)
    result = estimate_cross_fitted_trust(
        features,
        labels,
        config=_fast_config(rounds=4, convergence_tolerance=1e-8),
    )

    assert result.converged
    assert result.rounds_run == 1
    torch.testing.assert_close(result.suspicion, torch.full((24,), 0.5))
    torch.testing.assert_close(result.trust, torch.ones(24))


def test_api_cannot_receive_generator_metadata() -> None:
    parameters = set(inspect.signature(estimate_cross_fitted_trust).parameters)
    assert parameters == {
        "features",
        "labels",
        "config",
        "initial_trust",
        "fold_ids",
    }
    assert parameters.isdisjoint({"clean_labels", "minority", "flipped"})


def test_cross_fitted_dynamics_downtrusts_clear_label_noise() -> None:
    generator = torch.Generator().manual_seed(101)
    labels = torch.cat((torch.zeros(48), torch.ones(48)))
    features = torch.empty(96, 3)
    features[:, 0] = labels.mul(4).sub(2)
    features[:, 1:] = 0.3 * torch.randn(96, 2, generator=generator)
    flipped = torch.zeros(96, dtype=torch.bool)
    flipped[torch.tensor([3, 9, 18, 29, 37, 52, 61, 70, 82, 91])] = True
    observed = torch.where(flipped, 1 - labels, labels)

    result = estimate_cross_fitted_trust(
        features,
        observed,
        config=_fast_config(
            folds=4,
            rounds=3,
            training_steps=36,
            hidden_dimensions=6,
            max_delta=0.2,
        ),
    )

    assert result.trust[flipped].mean() + 0.2 < result.trust[~flipped].mean()
    assert result.suspicion[flipped].mean() > result.suspicion[~flipped].mean()
