# Examples

Runnable examples for the `symbolic_regression` Python library. Every script is
self-contained; comments and output are in English.

Install the project once from the repository root:

```bash
uv sync
```

## Scripts

| Script | What it shows |
|---|---|
| [`basic_fit.py`](basic_fit.py) | Fit `y = x**2` with the default offline policy and print the recovered expression and metrics. |
| [`operator_subset.py`](operator_subset.py) | Restrict the search to `add` and `mul` and verify that every decision only uses those operators. |
| [`multi_feature.py`](multi_feature.py) | Fit a two-feature dataset passed as a 2-D array, then predict with arrays and with named columns. |
| [`regressor_wrapper.py`](regressor_wrapper.py) | Use the stateful `SymbolicRegressor` wrapper with `metrics_`, `diagnostics_` and `predict`. |
| [`diagnostics.py`](diagnostics.py) | Inspect `to_dict()` with and without traces, plus the human-readable summary. |
| [`custom_policy.py`](custom_policy.py) | Plug in a custom operator policy implementing the `OperatorPolicy` protocol. |
| [`decision_model.py`](decision_model.py) | Ask a local Ollama decision model to rank operators (requires a server and a decision-capable model). |

Run any of them with:

```bash
uv run python examples/basic_fit.py
uv run python examples/operator_subset.py
uv run python examples/multi_feature.py
uv run python examples/regressor_wrapper.py
uv run python examples/diagnostics.py
uv run python examples/custom_policy.py
uv run python examples/decision_model.py
```

All examples except `decision_model.py` use the deterministic `offline` policy
and need no external services. `decision_model.py` exits cleanly with a message
when Ollama is not available.
