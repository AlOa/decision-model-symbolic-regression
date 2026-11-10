"""Deterministic dataset generation from the built-in function suite."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..data import Dataset
from .functions import FunctionSpec, get_function


class DatasetGenerationError(ValueError):
    """Raised when a synthetic dataset cannot be generated."""


def _resolve_domain(
    spec: FunctionSpec, domain: Mapping[str, tuple[float, float]] | None
) -> dict[str, tuple[float, float]]:
    resolved = dict(spec.domain)
    if domain is None:
        return resolved
    for name, bounds in domain.items():
        if name not in resolved:
            raise DatasetGenerationError(
                f"Domain override names unknown variable {name!r} for function {spec.name!r}"
            )
        low, high = bounds
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise DatasetGenerationError(
                f"Invalid domain override for {name!r}: ({low}, {high})"
            )
        resolved[name] = (float(low), float(high))
    return resolved


def generate_dataset(
    function: str | FunctionSpec,
    n_samples: int = 64,
    seed: int = 0,
    domain: Mapping[str, tuple[float, float]] | None = None,
    noise_std: float = 0.05,
    noise_snr_db: float | None = None,
) -> Dataset:
    """Generate a deterministic dataset for a suite function.

    Features are sampled uniformly from the declared domain. Noise is additive
    Gaussian and is applied only to the target. ``noise_std=0`` (or an SNR of
    infinity) reproduces the exact function values.
    """

    spec = get_function(function) if isinstance(function, str) else function
    if n_samples < 2:
        raise DatasetGenerationError("n_samples must be at least 2")
    if noise_std < 0:
        raise DatasetGenerationError("noise_std must be non-negative")
    if noise_snr_db is not None and not np.isfinite(noise_snr_db):
        raise DatasetGenerationError("noise_snr_db must be finite")
    resolved_domain = _resolve_domain(spec, domain)
    rng = np.random.default_rng(seed)
    features = {
        name: rng.uniform(low, high, size=n_samples)
        for name, (low, high) in resolved_domain.items()
    }
    exact = spec.evaluate(features)
    if exact.ndim != 1 or exact.size != n_samples:
        raise DatasetGenerationError(
            f"Function {spec.name!r} returned shape {exact.shape}, expected ({n_samples},)"
        )
    if not np.all(np.isfinite(exact)):
        raise DatasetGenerationError(f"Function {spec.name!r} produced non-finite values")
    if noise_snr_db is not None:
        signal_std = float(np.std(exact))
        sigma = 0.0 if signal_std == 0 else signal_std / (10.0 ** (float(noise_snr_db) / 20.0))
        noise_mode = "snr_db"
        noise_value = float(noise_snr_db)
    else:
        sigma = float(noise_std)
        noise_mode = "std"
        noise_value = float(noise_std)
    if sigma == 0:
        target = exact.copy()
    else:
        target = exact + rng.normal(0.0, sigma, size=n_samples)
    if not np.all(np.isfinite(target)):
        raise DatasetGenerationError(f"Function {spec.name!r} produced non-finite targets")
    metadata = {
        "source": "function",
        "function": spec.name,
        "expression": spec.expression,
        "tier": spec.tier,
        "variables": list(spec.variables),
        "domain": {name: [float(bounds[0]), float(bounds[1])] for name, bounds in resolved_domain.items()},
        "sample_count": int(n_samples),
        "seed": int(seed),
        "noise_mode": noise_mode,
        "noise_value": noise_value,
        "noise_sigma": float(sigma),
    }
    return Dataset.from_arrays(features, target, metadata=metadata)
