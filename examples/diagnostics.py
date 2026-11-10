"""Inspect the diagnostic payload returned by fit().

Run with:
    uv run python examples/diagnostics.py
"""

import json

import numpy as np

from symbolic_regression import fit


def main() -> None:
    x = np.linspace(-2.0, 2.0, 51)
    y = x**2 + 0.5 * x

    result = fit(
        {"x": x},
        y,
        operators=("mul", "add"),
        iterations=100,
        seed=0,
    )

    # A compact report without the per-iteration and per-decision traces.
    compact = result.to_dict(include_trace=False)
    print("Compact diagnostics:")
    print(json.dumps(compact, indent=2, sort_keys=True))

    # The full trace contains every search iteration and policy decision.
    full = result.to_dict(include_trace=True)
    print(f"\nIterations recorded: {len(full['history'])}")
    print(f"Decisions recorded:  {len(full['decisions'])}")

    if full["decisions"]:
        first = full["decisions"][0]
        print("\nFirst decision:")
        print(f"  expression:    {first['expression']}")
        print(f"  source:        {first['source']}")
        print(f"  candidates:    {', '.join(first['candidates'])}")
        print(f"  probabilities: {first['probabilities']}")

    print("\nHuman-readable summary:")
    print(result.summary())


if __name__ == "__main__":
    main()
