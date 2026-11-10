# SimbolicRegression

Decision-model-guided symbolic regression with Monte Carlo Tree Search.

The project builds symbolic expressions incrementally from a dataset. At each
step an Ollama decision model ranks candidate operators, the returned
probabilities become search priors, and Monte Carlo Tree Search (MCTS) uses the
data fit as its reward.

## Status

This repository contains a research prototype. See the OpenSpec change
`decision-model-symbolic-regression` for the proposal, specs, design, and task
breakdown.

## Documentation

- [Technical exposition](docs/TECHNICAL.md): architecture, data flow, decision
  policy, MCTS, context format, artifacts, and testing.
- [Article](docs/ARTICLE.md): a narrative explanation of the idea, the method,
  and what the prototype can be used for.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Optional: a running Ollama server with a decision-capable model such as
  `d3:4b`

## Development

```bash
uv sync
uv run pytest -q
```

## Quick start

```bash
# Run without Ollama using the deterministic offline policy.
uv run symbolic-regression run --function quadratic --policy offline \
  --iterations 20 --samples 64 --noise-std 0.05 --output runs --run-name demo

# Run with a local decision model.
uv run symbolic-regression run --function sine --model d3:4b \
  --iterations 50 --output runs --run-name live

# Compare the decision policy with a uniform baseline.
uv run symbolic-regression compare --function composite \
  --policies decision,uniform --iterations 50 --output runs --run-name compare

# Load a CSV dataset.
uv run symbolic-regression run --csv data.csv --target y --policy offline \
  --output runs --run-name csv

# Restrict the operators used by the search.
uv run symbolic-regression run --function quadratic --policy offline \
  --operators add,mul --iterations 20 --output runs --run-name restricted
```

## Python library

The package exposes a programmatic API for use inside another program:

```python
import numpy as np
from symbolic_regression import fit

x = np.linspace(-2.0, 2.0, 41)
result = fit(
    {"x": x},
    x**2,
    operators=("mul", "add", "sub"),
    iterations=50,
    seed=0,
)

print(result.expression_str)          # recovered tree, e.g. "(x * x)"
print(result.simplified_expression)   # SymPy form, e.g. "x**2"
print(result.r2, result.mse, result.nmse)
print(result.stop_reason.value)
print(result.to_dict())               # full diagnostics, including traces
```

`fit` accepts a mapping of column name to values, a 2-D NumPy array (with
`feature_names`), or a data-frame-like object. The target can be a values array
or a column name when the input container already contains it. Pass
`operators=...` to restrict the search (`None` means every registered operator;
`available_operators()` lists them). The default policy is `"offline"`; use
`policy="decision"` with a running Ollama server, or pass a custom policy
instance.

`FitResult` exposes `expression`, `expression_str`, `simplified_expression`,
`evaluation` (plus `mse`, `nmse`, `r2`, and `valid` properties), `reward`,
`steps`, `stop_reason`, `iterations`, `decision_calls`, `fallback_calls`,
`elapsed_seconds`, `history`, `decisions`, and `predict(features)`. There is
also a small stateful wrapper:

```python
from symbolic_regression import SymbolicRegressor

model = SymbolicRegressor(operators=("mul", "add"), iterations=50)
model.fit({"x": x}, x**2)
print(model.simplified_expression_)
print(model.diagnostics_)
predictions = model.predict({"x": np.array([-1.0, 0.0, 1.0])})
```

Runnable, self-contained examples are in [`examples/`](examples/README.md); they
cover basic fits, operator subsets, multi-feature data, diagnostics, a custom
policy, the regressor wrapper, and the Ollama decision policy.

A TOML configuration file can hold the same options; command-line values
override it:

```toml
[run]
function = "quadratic"
policy = "offline"
iterations = 20
samples = 64
noise_std = 0.05
output = "runs"
run_name = "demo"
```

```bash
uv run symbolic-regression run --config run.toml --iterations 5
```

Each run writes to `<output>/<run-name>/`:

- `result.json`: recovered expression, its SymPy-simplified form, ground truth,
  metrics, complexity, decision counts, stop reason, dataset metadata, and
  search settings.
- `trace.json`: per-decision records (probabilities, confidence, errors) and
  per-iteration search records.
- `summary.md`: human-readable summary.

`compare` also writes `comparison.json` and `comparison.md` with both policies
side by side. Runs with the same seed, configuration, and deterministic policy
produce identical artifacts apart from timestamps and durations.

## Function suite and noisy datasets

The built-in suite lives in `symbolic_regression.experiments.functions` and is
shared by tests and experiments.

| Tier | Functions | Example expressions | Default domain |
|---|---|---|---|
| 1 | `constant`, `linear` | `3.5`, `2*x - 1` | `x` in `[-3, 3]` |
| 2 | `quadratic`, `cubic` | `x**2 - 3*x + 2`, `x**3 - 2*x` | `x` in `[-3, 3]` (cubic: `[-2, 2]`) |
| 3 | `sine`, `damped_sine`, `exponential` | `sin(2*x)`, `exp(-x/5)*sin(3*x)`, `exp(x/3)` | `x` in `[-pi, pi]` (exponential: `[-3, 3]`) |
| 4 | `rational`, `log_abs`, `poly2d`, `composite` | `1/(1+x**2)`, `log(abs(x)+1)`, `x0**2 + x0*x1 + x1`, `x**2 + sin(3*x) + 0.5` | `x` in `[-3, 3]`; `poly2d`: `x0, x1` in `[-2, 2]` |

Datasets are generated as `y = f(X) + noise`. Noise is additive Gaussian and
is applied only to the target; features stay noise-free. Configure it with
`noise_std` (default `0.05`) or `noise_snr_db` (for example `20`). A zero noise
level reproduces the exact function values, and the same seed plus
configuration produces the same dataset.

```python
from symbolic_regression.experiments.datasets import generate_dataset

dataset = generate_dataset("sine", n_samples=32, seed=42, noise_std=0.0)
print(dataset.n_rows, dataset.metadata["function"], dataset.metadata["noise_mode"])
```

## Ollama decision models

The decision policy uses Ollama's native SystemOne endpoint instead of chat
prompting, so operator probabilities and confidence come from the model's
decision head.

Start the server and check for a decision-capable model:

```bash
ollama serve
curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; print([m['name'] for m in json.load(sys.stdin)['models'] if 'decision' in (m.get('capabilities') or [])])"
```

The default model is `d3:4b`. Configure another decision-capable model with
`--model`.

The client sends one `choice` question to `POST /v1/systemone`:

```json
{
  "model": "d3:4b",
  "state": {"step": 0, "expression": "x"},
  "questions": {
    "operator": {
      "type": "choice",
      "instructions": "choose",
      "criteria": {"sin": "sine", "cos": "cosine"}
    }
  }
}
```

A successful response looks like:

```json
{
  "model": "d3:4b",
  "answers": {
    "operator": {
      "type": "choice",
      "choice": "sin",
      "probabilities": {"sin": 0.7, "cos": 0.3},
      "confidence": 0.9
    }
  },
  "usage": {"input_tokens": 10, "output_tokens": 0}
}
```

If the server is unreachable, the model lacks the `decision` capability, or the
response is malformed, the policy reports the failure and can fall back to a
uniform distribution. Use `--policy offline` to run without any Ollama server.

## Search parameters and reward

MCTS is a PUCT search. The main options are:

| Option | Default | Meaning |
|---|---|---|
| `iterations` | `50` | maximum search iterations |
| `max_nodes` | `500` | maximum tree nodes |
| `max_depth` | `6` | maximum expression depth |
| `rollout_depth` | `3` | steps simulated from an expanded node |
| `c_puct` | `1.4` | exploration constant in the PUCT bound |
| `top_applications` | `2` | operand applications kept per operator |
| `max_operands` | `256` | size of the deterministic operand bank |
| `fit_threshold` | `0.995` | R-squared value that stops the search |
| `max_steps` | `12` | maximum construction steps in the result |
| `time_limit_seconds` | `300` | wall-clock budget |
| `greedy` | `false` | follow only the highest-probability operator path |

The selection bound is:

```text
Q(child) + c_puct * P(child) * sqrt(N(parent)) / (1 + N(child))
```

The reward of an evaluated expression is `1 / (1 + NMSE)`, so a perfect fit
gives `1` and invalid or non-finite evaluations give `0`. The final expression
is the best feasible result by reward; ties are broken by lower node count and
then by deterministic expression order.



