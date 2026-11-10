"""Show how the operator set constrains the search.

Run with:
    uv run python examples/operator_subset.py

The target is y = x**2 - 3*x + 2. Only "add" and "mul" are allowed, which is
enough for this polynomial. Every decision made by the offline policy can only
choose between those two operators.
"""

import numpy as np

from symbolic_regression import fit


def main() -> None:
    x = np.linspace(-3.0, 3.0, 61)
    y = x**2 - 3.0 * x + 2.0

    result = fit(
        {"x": x},
        y,
        operators=("add", "mul"),
        iterations=100,
        seed=0,
    )

    print(f"Allowed operators:     {', '.join(result.operators)}")
    print(f"Raw expression:        {result.expression_str}")
    print(f"Simplified expression: {result.simplified_expression}")
    print(f"R-squared:             {result.r2:.6f}")

    # The trace records which candidates the policy could choose from.
    candidates = sorted({name for record in result.decisions for name in record["candidates"]})
    print(f"Policy candidates:     {', '.join(candidates)}")

    # Every expanded operator must belong to the requested subset.
    expanded = {record["expanded_operator"] for record in result.history}
    expanded.discard(None)
    print(f"Expanded operators:    {', '.join(sorted(expanded))}")


if __name__ == "__main__":
    main()
