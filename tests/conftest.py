import pytest

from symbolic_regression.experiments.datasets import generate_dataset
from symbolic_regression.experiments.functions import SUITE

NOISE_LEVELS = [("zero", 0.0), ("low", 0.05), ("high", 0.1)]


@pytest.fixture(
    params=[
        (name, label, noise_std)
        for name in SUITE
        for label, noise_std in NOISE_LEVELS
    ],
    ids=[f"{name}-{label}" for name in SUITE for label, _ in NOISE_LEVELS],
)
def suite_dataset(request):
    name, label, noise_std = request.param
    dataset = generate_dataset(name, n_samples=32, seed=123, noise_std=noise_std)
    return name, label, dataset
