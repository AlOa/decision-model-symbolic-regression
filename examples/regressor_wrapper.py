"""Use the scikit-learn-style SymbolicRegressor wrapper.

Run with:
    uv run python examples/regressor_wrapper.py
"""

import numpy as np

from symbolic_regression import SymbolicRegressor


def main() -> None:
    x = np.linspace(-2.0, 2.0, 41)
    y = x**2 - 2.0 * x + 1.0

    # Search options provided here are reused by every call to fit().
    model = SymbolicRegressor(
        operators=("mul", "add", "sub"),
        iterations=100,
        seed=0,
    )

    # Fitting returns the wrapper itself, so calls can be chained.
    returned = model.fit({"x": x}, y)
    assert returned is model

    print(f"Raw expression:        {model.expression_str_}")
    print(f"Simplified expression: {model.simplified_expression_}")
    print(f"Metrics:               {model.metrics_}")

    diagnostics = model.diagnostics_
    print(f"Stop reason:           {diagnostics['stop_reason']}")
    print(f"Decision calls:        {diagnostics['decision_calls']}")

    predictions = model.predict({"x": np.array([-1.0, 0.0, 1.0, 2.0])})
    print(f"Predictions:           {predictions}")


if __name__ == "__main__":
    main()
