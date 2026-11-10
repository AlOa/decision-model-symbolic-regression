"""Fit a two-feature dataset passed as a 2-D NumPy array.

Run with:
    uv run python examples/multi_feature.py

The target is y = x0*x1 + 0.5*x0. Feature names are supplied explicitly so
the recovered expression and predictions use meaningful symbols.
"""

import numpy as np

from symbolic_regression import fit


def main() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-2.0, 2.0, 80)
    x1 = rng.uniform(-2.0, 2.0, 80)
    y = x0 * x1 + 0.5 * x0

    features = np.column_stack([x0, x1])
    result = fit(
        features,
        y,
        feature_names=["x0", "x1"],
        operators=("mul", "add"),
        iterations=100,
        seed=1,
    )

    print(f"Raw expression:        {result.expression_str}")
    print(f"Simplified expression: {result.simplified_expression}")
    print(f"Features:              {', '.join(result.feature_names)}")
    print(f"R-squared:             {result.r2:.6f}")

    # Predict on new rows using the same 2-D array layout.
    new_rows = np.array(
        [
            [1.0, 2.0],
            [2.0, -1.0],
            [0.5, 0.5],
        ]
    )
    predictions = result.predict(new_rows)
    print("\nPredictions:")
    for (row_x0, row_x1), prediction in zip(new_rows, predictions):
        print(f"  x0={row_x0:+.1f}, x1={row_x1:+.1f} -> {prediction:+.4f}")

    # Alternatively, pass columns by name.
    print("\nPrediction from a mapping:", result.predict({"x0": [1.0], "x1": [2.0]}))


if __name__ == "__main__":
    main()
