import numpy as np
import pytest

from symbolic_regression.experiments.datasets import DatasetGenerationError, generate_dataset
from symbolic_regression.experiments.functions import get_function


@pytest.mark.parametrize("name", ["constant", "linear", "quadratic", "sine", "rational", "poly2d", "composite"])
def test_zero_noise_is_exact(name: str) -> None:
    spec = get_function(name)
    dataset = generate_dataset(name, n_samples=32, seed=11, noise_std=0.0)
    exact = spec.evaluate(dataset.features)
    np.testing.assert_allclose(dataset.target, exact)
    assert dataset.metadata["noise_sigma"] == 0.0


def test_noise_affects_only_the_target() -> None:
    clean = generate_dataset("quadratic", n_samples=64, seed=3, noise_std=0.0)
    noisy = generate_dataset("quadratic", n_samples=64, seed=3, noise_std=0.1)
    for name in clean.feature_names:
        np.testing.assert_allclose(clean.features[name], noisy.features[name])
    assert not np.allclose(clean.target, noisy.target)


def test_noisy_dataset_is_reproducible() -> None:
    first = generate_dataset("composite", n_samples=64, seed=9, noise_std=0.05)
    second = generate_dataset("composite", n_samples=64, seed=9, noise_std=0.05)
    for name in first.feature_names:
        np.testing.assert_allclose(first.features[name], second.features[name])
    np.testing.assert_allclose(first.target, second.target)
    assert first.metadata == second.metadata


def test_two_noise_levels_share_features_but_differ_in_targets() -> None:
    low = generate_dataset("sine", n_samples=64, seed=4, noise_std=0.05)
    high = generate_dataset("sine", n_samples=64, seed=4, noise_std=0.2)
    for name in low.feature_names:
        np.testing.assert_allclose(low.features[name], high.features[name])
    assert not np.allclose(low.target, high.target)


def test_snr_mode_is_recorded() -> None:
    dataset = generate_dataset("linear", n_samples=32, seed=1, noise_snr_db=20.0)
    assert dataset.metadata["noise_mode"] == "snr_db"
    assert dataset.metadata["noise_value"] == 20.0
    assert dataset.metadata["noise_sigma"] > 0


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(DatasetGenerationError):
        generate_dataset("linear", n_samples=1)
    with pytest.raises(DatasetGenerationError):
        generate_dataset("linear", noise_std=-1.0)
    with pytest.raises(DatasetGenerationError):
        generate_dataset("linear", domain={"missing": (0.0, 1.0)})
