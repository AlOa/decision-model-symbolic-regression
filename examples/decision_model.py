"""Use a local Ollama decision model as the operator policy.

Run with:
    uv run python examples/decision_model.py

This example requires a running Ollama server and a decision-capable model.
The default model name is "d3:4b"; pass another one via the MODEL constant.
Without a server the script reports the error and exits cleanly.
"""

import numpy as np

from symbolic_regression import fit
from symbolic_regression.decision.client import OllamaError

MODEL = "d3:4b"
BASE_URL = "http://localhost:11434"


def main() -> None:
    x = np.linspace(-3.0, 3.0, 81)
    y = np.sin(2.0 * x)

    try:
        result = fit(
            {"x": x},
            y,
            # "decision" asks the model to rank operators at every step.
            policy="decision",
            model=MODEL,
            base_url=BASE_URL,
            iterations=50,
            seed=0,
        )
    except OllamaError as exc:
        print("The decision policy is unavailable:")
        print(f"  {exc}")
        print("Start Ollama and pull a decision-capable model, or run:")
        print("  uv run python examples/basic_fit.py")
        return

    print(f"Raw expression:        {result.expression_str}")
    print(f"Simplified expression: {result.simplified_expression}")
    print(f"R-squared:             {result.r2:.6f}")
    print(f"Decision calls:        {result.decision_calls}")
    print(f"Fallback calls:        {result.fallback_calls}")

    sources = {record["source"] for record in result.decisions}
    print(f"Decision sources:      {', '.join(sorted(sources))}")


if __name__ == "__main__":
    main()
