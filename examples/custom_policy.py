"""Plug a custom operator policy into the search.

Run with:
    uv run python examples/custom_policy.py

A policy only has to expose ``name`` and a ``probabilities(context)`` method
returning a PolicyResult. This example always prefers multiplication over
addition, which is enough to recover y = x**2.
"""

import numpy as np

from symbolic_regression import fit
from symbolic_regression.decision.policy import PolicyResult


class MultiplicationFirst:
    """Deterministic policy with fixed operator probabilities."""

    name = "multiplication-first"

    def probabilities(self, context) -> PolicyResult:
        probabilities = {"mul": 0.8, "add": 0.2}
        return PolicyResult(
            probabilities=probabilities,
            selected="mul",
            source="custom",
        )


def main() -> None:
    x = np.linspace(-2.0, 2.0, 41)
    y = x**2

    result = fit(
        {"x": x},
        y,
        operators=("mul", "add"),
        policy=MultiplicationFirst(),
        iterations=20,
        seed=0,
    )

    print(f"Policy:                {result.policy}")
    print(f"Raw expression:        {result.expression_str}")
    print(f"Simplified expression: {result.simplified_expression}")
    print(f"R-squared:             {result.r2:.6f}")


if __name__ == "__main__":
    main()
