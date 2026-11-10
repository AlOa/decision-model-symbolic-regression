import numpy as np
import pytest

from symbolic_regression.experiments.datasets import generate_dataset
from symbolic_regression.experiments.functions import SUITE, get_function, list_functions, tiers


def test_all_complexity_tiers_are_represented() -> None:
    by_tier = tiers()
    assert set(by_tier) == {1, 2, 3, 4}
    assert all(by_tier[tier] for tier in by_tier)


def test_function_metadata_is_complete() -> None:
    for spec in list_functions():
        assert spec.name
        assert spec.expression
        assert spec.variables
        assert spec.tier in (1, 2, 3, 4)
        assert spec.domain
        assert set(spec.variables) <= set(spec.domain)
        for low, high in spec.domain.values():
            assert np.isfinite(low) and np.isfinite(high) and low < high


def test_unknown_function_is_reported() -> None:
    with pytest.raises(KeyError, match="Unknown function"):
        get_function("does-not-exist")


@pytest.mark.parametrize("name", list(SUITE))
def test_suite_functions_produce_finite_samples_within_domain(name: str) -> None:
    spec = get_function(name)
    dataset = generate_dataset(name, n_samples=48, seed=5, noise_std=0.0)
    for variable in spec.variables:
        low, high = spec.domain[variable]
        assert np.all(dataset.features[variable] >= low)
        assert np.all(dataset.features[variable] <= high)
    assert np.all(np.isfinite(dataset.target))
    assert dataset.metadata["function"] == name
    assert dataset.metadata["tier"] == spec.tier
    assert dataset.metadata["variables"] == list(spec.variables)
    assert dataset.metadata["domain"]
