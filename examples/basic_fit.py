"""Fit a simple quadratic function with the default offline policy.

Run with:
    uv run python examples/basic_fit.py
"""

import numpy as np

from symbolic_regression import fit


def main() -> None:
    # Generate a small noise-free dataset for y = x**2.
    x = np.linspace(-2.0, 2.0, 41)
    y = x**2

    result = fit(
        {"x": x},
        y,
        # Restricting the operator set keeps this example small and fast.
        operators=("mul", "add", "sub"),
        iterations=50,
        seed=0,
    )

    print(f"Raw expression:        {result.expression_str}")
    print(f"Simplified expression: {result.simplified_expression}")
    print(f"R-squared:             {result.r2:.6f}")
    print(f"Stop reason:           {result.stop_reason.value}")
    print(f"Decision calls:        {result.decision_calls}")


if __name__ == "__main__":
    main()
