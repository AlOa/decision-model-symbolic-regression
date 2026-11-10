"""Built-in ground-truth function suite with complexity tiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True, slots=True)
class FunctionSpec:
    name: str
    expression: str
    function: Callable[..., np.ndarray]
    variables: tuple[str, ...]
    tier: int
    domain: dict[str, tuple[float, float]]
    constraints: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Function name must not be empty")
        if self.tier not in (1, 2, 3, 4):
            raise ValueError(f"Function {self.name!r} has unsupported tier {self.tier}")
        if not self.variables:
            raise ValueError(f"Function {self.name!r} must declare at least one variable")
        missing = [name for name in self.variables if name not in self.domain]
        if missing:
            raise ValueError(
                f"Function {self.name!r} is missing domains for: {', '.join(missing)}"
            )
        for name, bounds in self.domain.items():
            low, high = bounds
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise ValueError(f"Function {self.name!r} has an invalid domain for {name!r}")

    def evaluate(self, features: dict[str, np.ndarray]) -> np.ndarray:
        missing = [name for name in self.variables if name not in features]
        if missing:
            raise ValueError(f"Missing variables for {self.name!r}: {', '.join(missing)}")
        return np.asarray(self.function(*[features[name] for name in self.variables]), dtype=float)


def _constant(x: np.ndarray) -> np.ndarray:
    return np.full_like(np.asarray(x, dtype=float), 3.5, dtype=float)


SUITE: dict[str, FunctionSpec] = {
    "constant": FunctionSpec(
        name="constant",
        expression="3.5",
        function=_constant,
        variables=("x",),
        tier=1,
        domain={"x": (-3.0, 3.0)},
        description="constant function",
    ),
    "linear": FunctionSpec(
        name="linear",
        expression="2*x - 1",
        function=lambda x: 2.0 * x - 1.0,
        variables=("x",),
        tier=1,
        domain={"x": (-3.0, 3.0)},
        description="affine function",
    ),
    "quadratic": FunctionSpec(
        name="quadratic",
        expression="x**2 - 3*x + 2",
        function=lambda x: x**2 - 3.0 * x + 2.0,
        variables=("x",),
        tier=2,
        domain={"x": (-3.0, 3.0)},
        description="quadratic polynomial",
    ),
    "cubic": FunctionSpec(
        name="cubic",
        expression="x**3 - 2*x",
        function=lambda x: x**3 - 2.0 * x,
        variables=("x",),
        tier=2,
        domain={"x": (-2.0, 2.0)},
        description="cubic polynomial",
    ),
    "sine": FunctionSpec(
        name="sine",
        expression="sin(2*x)",
        function=lambda x: np.sin(2.0 * x),
        variables=("x",),
        tier=3,
        domain={"x": (-np.pi, np.pi)},
        description="trigonometric function",
    ),
    "damped_sine": FunctionSpec(
        name="damped_sine",
        expression="exp(-x/5) * sin(3*x)",
        function=lambda x: np.exp(-x / 5.0) * np.sin(3.0 * x),
        variables=("x",),
        tier=3,
        domain={"x": (-np.pi, np.pi)},
        description="damped trigonometric function",
    ),
    "exponential": FunctionSpec(
        name="exponential",
        expression="exp(x/3)",
        function=lambda x: np.exp(x / 3.0),
        variables=("x",),
        tier=3,
        domain={"x": (-3.0, 3.0)},
        description="exponential function",
    ),
    "rational": FunctionSpec(
        name="rational",
        expression="1/(1 + x**2)",
        function=lambda x: 1.0 / (1.0 + x**2),
        variables=("x",),
        tier=4,
        domain={"x": (-3.0, 3.0)},
        constraints="x is real; denominator is always positive",
        description="rational function",
    ),
    "log_abs": FunctionSpec(
        name="log_abs",
        expression="log(abs(x) + 1)",
        function=lambda x: np.log(np.abs(x) + 1.0),
        variables=("x",),
        tier=4,
        domain={"x": (-3.0, 3.0)},
        constraints="abs(x) + 1 is always positive",
        description="logarithmic function",
    ),
    "poly2d": FunctionSpec(
        name="poly2d",
        expression="x0**2 + x0*x1 + x1",
        function=lambda x0, x1: x0**2 + x0 * x1 + x1,
        variables=("x0", "x1"),
        tier=4,
        domain={"x0": (-2.0, 2.0), "x1": (-2.0, 2.0)},
        description="multivariate polynomial",
    ),
    "composite": FunctionSpec(
        name="composite",
        expression="x**2 + sin(3*x) + 0.5",
        function=lambda x: x**2 + np.sin(3.0 * x) + 0.5,
        variables=("x",),
        tier=4,
        domain={"x": (-3.0, 3.0)},
        description="composite polynomial and trigonometric function",
    ),
}


def get_function(name: str) -> FunctionSpec:
    try:
        return SUITE[name]
    except KeyError as exc:
        available = ", ".join(SUITE)
        raise KeyError(f"Unknown function {name!r}. Available functions: {available}") from exc


def list_functions() -> list[FunctionSpec]:
    return list(SUITE.values())


def tiers() -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for spec in SUITE.values():
        result.setdefault(spec.tier, []).append(spec.name)
    return result
