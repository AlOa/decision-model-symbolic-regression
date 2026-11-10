# Usage Guide

This guide explains how to use `symbolic_regression` as a library and through
the `symbolic-regression` CLI. For an overview and installation instructions,
see the [README](../README.md).

## 1. Installation

From a clone of the repository:

```bash
uv sync
```

Once the package is published on PyPI:

```bash
uv add decision-model-symbolic-regression
# or
pip install decision-model-symbolic-regression
```

The import name is always `symbolic_regression`.

## 2. Quick library example

```python
import numpy as np
from symbolic_regression import fit

x = np.linspace(-2.0, 2.0, 41)
result = fit({"x": x}, x**2, operators=("mul", "add"), seed=0)

print(result.expression_str)
print(result.simplified_expression)
print(result.r2)
```

## 3. Data inputs

`fit(features, target, ...)` accepts the following forms.

### Mapping of column name to values

```python
fit({"x0": x0, "x1": x1}, y)
```

### Mapping that contains the target column

```python
fit({"x": x, "y": y}, "y")
```

When `target` is a string and `features` is a mapping, that key is used as the
target and excluded from the features.

### 2-D NumPy array

```python
X = np.column_stack([x0, x1])
fit(X, y, feature_names=["x0", "x1"])
```

Without `feature_names`, a single column is called `x`; multiple columns are
called `x0`, `x1`, and so on.

### Data-frame-like object

Any object exposing `columns` and column indexing works, including pandas and
polars frames:

```python
fit(df, "y")                       # target column by name
fit(df[["x0", "x1"]], y)           # target as an array
```

A data frame may include the target column when `target` is its name; that column
is excluded from the features.

### A 1-D feature array

```python
fit(x, y)  # one feature named "x"
```

### Prediction

`FitResult.predict` accepts the same input shapes:

```python
result.predict({"x": [1.0, 2.0]})
result.predict(np.array([[1.0, 2.0]]))     # for two features
result.predict(df[["x0", "x1"]])
```

## 4. Operators

The registered operators are:

| Category | Names |
|---|---|
| Binary | `add`, `sub`, `mul`, `div` |
| Unary | `sin`, `cos`, `exp`, `log`, `sqrt`, `abs` |

`available_operators()` returns the full list.

Restrict the search with `operators`:

```python
fit(X, y, operators=("add", "mul"))
fit(X, y, operators="add,mul")        # comma-separated form
```

Restriction affects the policy candidates, the operand bank, and the
applications allowed in the tree. Compound operands are only created when their
binary operator is allowed.

## 5. Policies

| Policy | Needs Ollama | Deterministic | Use case |
|---|---|---|---|
| `"offline"` (default) | no | yes, for a fixed seed | development, tests, reproducible demos |
| `"uniform"` | no | yes | baseline without priors |
| `"decision"` | yes | model-dependent | model-guided operator priors |
| custom object | depends | depends | experiments with your own priors |

### Decision policy

```python
result = fit(
    {"x": x},
    y,
    policy="decision",
    model="d3:4b",
    base_url="http://localhost:11434",
    iterations=50,
)
```

`fit` checks model readiness before searching. Without a server, use
`policy="offline"`.

### Custom policy

A policy is any object with a `name` attribute and a
`probabilities(context)` method returning a `PolicyResult`:

```python
from symbolic_regression import fit
from symbolic_regression.decision.policy import PolicyResult


class MultiplicationFirst:
    name = "multiplication-first"

    def probabilities(self, context) -> PolicyResult:
        return PolicyResult(
            probabilities={"mul": 0.8, "add": 0.2},
            selected="mul",
            source="custom",
        )


result = fit({"x": x}, y, operators=("mul", "add"), policy=MultiplicationFirst())
```

If a custom policy returns an operator outside the allowed set, that operator is
ignored by the search.

## 6. `fit` parameters

`fit` combines data handling, policy construction, and MCTS configuration.

### Data and expression options

| Parameter | Default | Meaning |
|---|---|---|
| `feature_names` | inferred | names for 2-D array columns |
| `operators` | all | allowed operator names |
| `initial_expression` | first feature | starting expression node or feature name |

### Policy options

| Parameter | Default | Meaning |
|---|---|---|
| `policy` | `"offline"` | `"offline"`, `"uniform"`, `"decision"`, or a custom object |
| `model` | `"d3:4b"` | Ollama decision model |
| `base_url` | `"http://localhost:11434"` | Ollama base URL |
| `request_timeout` | `30.0` | request timeout in seconds |
| `request_retries` | `1` | retries per request |
| `client` | `None` | reuse an existing `OllamaClient` |

### Search options

| Parameter | Default | Meaning |
|---|---|---|
| `seed` | `0` | seed for the offline policy and context sampling |
| `iterations` | `50` | maximum MCTS iterations |
| `max_nodes` | `500` | maximum tree nodes |
| `max_depth` | `6` | maximum expression depth |
| `rollout_depth` | `3` | simulated steps from an expanded node |
| `c_puct` | `1.4` | exploration constant |
| `top_applications` | `2` | operand applications kept per operator |
| `max_operands` | `256` | operand bank size |
| `fit_threshold` | `0.995` | R-squared that stops the search |
| `max_steps` | `12` | maximum construction steps in the result |
| `time_limit_seconds` | `300.0` | wall-clock budget |
| `greedy` | `False` | follow only the best operator path |
| `context_rows` | `24` | rows sampled for the decision context |
| `residual_values` | `16` | residuals sampled for the decision context |

## 7. `FitResult`

| Attribute | Meaning |
|---|---|
| `expression` | expression tree (`Expr`) or `None` |
| `expression_str` | infix rendering with explicit parentheses |
| `simplified_expression` | best-effort SymPy rendering |
| `evaluation` | full `Evaluation` with predictions and residuals |
| `valid` | whether a feasible expression was found |
| `mse`, `nmse`, `r2` | fit metrics |
| `reward` | `1 / (1 + NMSE)` |
| `steps` | construction depth of the best expression |
| `stop_reason` | why the search stopped |
| `iterations` | iterations actually performed |
| `decision_calls` | policy calls |
| `fallback_calls` | policy calls that used the fallback distribution |
| `elapsed_seconds` | wall-clock search time |
| `feature_names` | feature names used by the model |
| `n_rows` | number of rows in the training data |
| `operators` | allowed operator names |
| `policy` | policy name |
| `history` | per-iteration search records |
| `decisions` | per-decision policy records |

### Serialization

```python
compact = result.to_dict(include_trace=False)   # metrics and settings only
full = result.to_dict()                         # also history and decisions

import json
print(json.dumps(full, indent=2, sort_keys=True))
```

### Summary

```python
print(result.summary())
```

## 8. `SymbolicRegressor`

`SymbolicRegressor` is a lightweight stateful wrapper:

```python
from symbolic_regression import SymbolicRegressor

model = SymbolicRegressor(
    operators=("mul", "add"),
    policy="offline",
    iterations=50,
    seed=0,
)
model.fit({"x": x}, y)

print(model.expression_str_)
print(model.simplified_expression_)
print(model.metrics_)
print(model.diagnostics_)
print(model.predict({"x": [1.0, 2.0]}))
```

Values passed to `SymbolicRegressor.fit` override constructor options.

## 9. CLI reference

```text
symbolic-regression run [options]
symbolic-regression compare [options]
```

### Data options

```text
--function NAME       built-in suite function
--csv PATH            CSV dataset
--target NAME         target column
--features LIST       comma-separated feature columns
--delimiter CHAR      CSV delimiter
--header / --no-header
```

### Policy options

```text
--policy NAME         offline, uniform, or decision
--policies LIST       policies for compare
--model NAME          Ollama decision model
--base-url URL        Ollama base URL
--request-timeout S   request timeout
--request-retries N   request retries
```

### Expression and search options

```text
--operators LIST      comma-separated allowed operators
--seed N
--samples N
--noise-std FLOAT
--noise-snr-db FLOAT
--iterations N
--max-nodes N
--max-depth N
--rollout-depth N
--c-puct FLOAT
--top-applications N
--max-operands N
--fit-threshold FLOAT
--max-steps N
--time-limit SECONDS
--greedy
--context-rows N
--residual-values N
```

### Output options

```text
--output DIR
--run-name NAME
--config PATH
```

The CLI exits with `0` on success and `2` on configuration or runtime errors.

## 10. Artifacts

A run writes to `<output>/<run-name>/`:

### `result.json`

- `run`: mode, policy, model, seed, function or CSV, output, timestamp, duration;
- `search`: search settings, including the operator subset;
- `dataset`: row count, feature names, metadata, and ground truth when available;
- `result`: expressions, metrics, reward, complexity, decision counts, stop
  reason, and duration.

### `trace.json`

- `decisions`: one record per policy decision;
- `iterations`: one record per MCTS iteration;
- `stop_reason`.

### `summary.md`

A human-readable summary of the run.

### Comparison

`compare` writes `comparison.json` and `comparison.md` with the policies side by
side.

## 11. Recipes

### Reproducible offline run

```python
result = fit(
    {"x": x},
    y,
    operators=("add", "mul", "sin"),
    policy="offline",
    seed=42,
    iterations=100,
)
```

The same data, seed, and configuration produce the same expression.

### Noisy data

```python
from symbolic_regression.experiments.datasets import generate_dataset

dataset = generate_dataset("sine", n_samples=64, seed=7, noise_std=0.05)
result = fit(dataset.features, dataset.target, operators=("add", "mul", "sin"))
```

### Small, fast search

```python
result = fit(
    {"x": x},
    y,
    operators=("add", "mul"),
    iterations=10,
    max_nodes=50,
    max_depth=3,
    rollout_depth=0,
    max_steps=3,
)
```

### Save diagnostics

```python
import json

with open("diagnostics.json", "w", encoding="utf-8") as handle:
    json.dump(result.to_dict(), handle, indent=2, sort_keys=True)
```

## 12. Troubleshooting

### `Unknown operator`

Use `available_operators()` to list valid names. Operators are case-sensitive.

### `The decision policy needs at least two operators`

A decision model must choose between at least two candidates. Add another
operator, or use `"offline"` / `"uniform"`.

### Ollama connection or model errors

- check that `ollama serve` is running;
- check the model name with `ollama list`;
- verify that the model advertises the `decision` capability;
- use `policy="offline"` to continue without a model.

### Search returns a poor expression

- increase `iterations`, `max_nodes`, or `max_steps`;
- allow more operators;
- clean the data or reduce noise;
- try a different seed and compare runs.

## 13. Reproducibility

- use an explicit `seed`;
- prefer `offline` or `uniform` for deterministic runs;
- pin the decision model and Ollama version when using `decision`;
- compare artifacts after removing `created_at` and `elapsed_seconds`.

## 14. License

MIT. See [LICENSE](../LICENSE).
