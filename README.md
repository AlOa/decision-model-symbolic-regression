# Decision-Model-Guided Symbolic Regression

[![tests](https://github.com/AlOa/decision-model-symbolic-regression/actions/workflows/tests.yml/badge.svg)](https://github.com/AlOa/decision-model-symbolic-regression/actions/workflows/tests.yml)

A research prototype that builds symbolic expressions incrementally from data.
At each step a decision model ranks candidate operators, the returned
probabilities become search priors, and Monte Carlo Tree Search (MCTS) uses the
data fit as its reward.

The project is a Python library and a small CLI. The library is the primary
interface: pass features, a target, and optionally a restricted operator set,
and get back the recovered expression plus diagnostics.

## Status

This is an **alpha research prototype**, not a production regression tool. It is
useful for experiments, teaching, and prototyping with decision-capable models.
It does not aim to replace mature symbolic-regression systems on large
benchmarks.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for development
- Optional: a running [Ollama](https://ollama.com/) server with a
  decision-capable model for the `decision` policy

## Installation

From source:

```bash
git clone https://github.com/AlOa/decision-model-symbolic-regression.git
cd decision-model-symbolic-regression
uv sync
uv run pytest -q
```

Once published on PyPI:

```bash
uv add decision-model-symbolic-regression
# or
pip install decision-model-symbolic-regression
```

The import name remains `symbolic_regression`:

```python
from symbolic_regression import fit
```

## Quick start

The offline policy is deterministic and needs no external server:

```bash
uv run symbolic-regression run --function quadratic --policy offline \
  --iterations 20 --samples 64 --noise-std 0.05 --output runs --run-name demo
```

Use a local decision model:

```bash
uv run symbolic-regression run --function sine --model d3:4b \
  --iterations 50 --output runs --run-name live
```

Compare the decision policy with a uniform baseline:

```bash
uv run symbolic-regression compare --function composite \
  --policies decision,uniform --iterations 50 --output runs --run-name compare
```

Load a CSV dataset:

```bash
uv run symbolic-regression run --csv data.csv --target y --policy offline \
  --output runs --run-name csv
```

Restrict the operator set:

```bash
uv run symbolic-regression run --function quadratic --policy offline \
  --operators add,mul --iterations 20 --output runs --run-name restricted
```

## Python library

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

`fit` accepts:

- a mapping of column name to 1-D values;
- a 2-D NumPy array, optionally with `feature_names`;
- a data-frame-like object exposing `columns` and column indexing (pandas,
  polars, or a lightweight custom object).

The target can be a values array, or a column name when the container already
contains it:

```python
result = fit({"x": x, "y": y}, "y", operators=("mul", "add"))
```

Use `operators=...` to restrict the search (`None` means every registered
operator; `available_operators()` lists them). The default policy is
`"offline"`; use `policy="decision"` with a running Ollama server, or pass a
custom policy object.

### Result and diagnostics

`FitResult` exposes:

- `expression`, `expression_str`, `simplified_expression`;
- `evaluation` and the `mse`, `nmse`, `r2`, `valid` properties;
- `reward`, `steps`, `stop_reason`, `iterations`;
- `decision_calls`, `fallback_calls`, `elapsed_seconds`;
- `history` and `decisions` (per-iteration and per-decision traces);
- `predict(features)`, `to_dict(...)`, and `summary()`.

There is also a small stateful wrapper:

```python
from symbolic_regression import SymbolicRegressor

model = SymbolicRegressor(operators=("mul", "add"), iterations=50)
model.fit({"x": x}, x**2)

print(model.expression_str_)
print(model.simplified_expression_)
print(model.metrics_)
print(model.diagnostics_)
predictions = model.predict({"x": np.array([-1.0, 0.0, 1.0])})
```

See [docs/USAGE.md](docs/USAGE.md) for the full guide: data formats, operator
sets, policies, `fit` parameters, result fields, CLI reference, artifacts, and
recipes.

## Examples

Runnable, self-contained scripts live in [`examples/`](examples/README.md):

- basic fit and metrics;
- operator restriction;
- multi-feature data and predictions;
- diagnostics and JSON export;
- custom operator policy;
- `SymbolicRegressor` wrapper;
- Ollama decision policy.

```bash
uv run python examples/basic_fit.py
uv run python examples/diagnostics.py
```

## Function suite and noisy datasets

The built-in suite lives in `symbolic_regression.experiments.functions` and is
shared by tests and experiments.

| Tier | Functions | Example expressions | Default domain |
|---|---|---|---|
| 1 | `constant`, `linear` | `3.5`, `2*x - 1` | `x` in `[-3, 3]` |
| 2 | `quadratic`, `cubic` | `x**2 - 3*x + 2`, `x**3 - 2*x` | `x` in `[-3, 3]` (cubic: `[-2, 2]`) |
| 3 | `sine`, `damped_sine`, `exponential` | `sin(2*x)`, `exp(-x/5)*sin(3*x)`, `exp(x/3)` | `x` in `[-pi, pi]` (exponential: `[-3, 3]`) |
| 4 | `rational`, `log_abs`, `poly2d`, `composite` | `1/(1+x**2)`, `log(abs(x)+1)`, `x0**2 + x0*x1 + x1`, `x**2 + sin(3*x) + 0.5` | `x` in `[-3, 3]`; `poly2d`: `x0, x1` in `[-2, 2]` |

Datasets are generated as `y = f(X) + noise`. Noise is additive Gaussian and is
applied only to the target; features stay noise-free. Configure it with
`noise_std` (default `0.05`) or `noise_snr_db` (for example `20`). A zero noise
level reproduces the exact function values, and the same seed plus configuration
produces the same dataset.

```python
from symbolic_regression.experiments.datasets import generate_dataset

dataset = generate_dataset("sine", n_samples=32, seed=42, noise_std=0.0)
print(dataset.n_rows, dataset.metadata["function"], dataset.metadata["noise_mode"])
```

## Ollama decision models

The `decision` policy uses Ollama's native SystemOne endpoint instead of chat
prompting, so operator probabilities and confidence come from the model's
decision head.

Start the server and check for a decision-capable model:

```bash
ollama serve
curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; print([m['name'] for m in json.load(sys.stdin)['models'] if 'decision' in (m.get('capabilities') or [])])"
```

The default model name is `d3:4b`. Replace it with any model that advertises the
`decision` capability and exposes the SystemOne endpoint:

```bash
uv run symbolic-regression run --function sine --model <your-model> --policy decision
```

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

> **Reproducibility note:** the decision policy depends on the exact model and
> Ollama build. The offline and uniform policies are deterministic for a fixed
> seed and configuration; the decision policy may not be. Pin and document the
> model you use if you publish results.

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
| `operators` | all | optional subset of operators |
| `context_rows` | `24` | rows sampled for the decision context |
| `residual_values` | `16` | residuals sampled for the decision context |

The selection bound is:

```text
Q(child) + c_puct * P(child) * sqrt(N(parent)) / (1 + N(child))
```

The reward of an evaluated expression is `1 / (1 + NMSE)`, so a perfect fit gives
`1` and invalid or non-finite evaluations give `0`. The final expression is the
best feasible result by reward; ties are broken by lower node count and then by
deterministic expression order.

## Run artifacts

Each CLI run writes to `<output>/<run-name>/`:

- `result.json`: recovered expression, its SymPy-simplified form, ground truth,
  metrics, complexity, decision counts, stop reason, dataset metadata, and
  search settings.
- `trace.json`: per-decision records (probabilities, confidence, errors) and
  per-iteration search records.
- `summary.md`: human-readable summary.

`compare` also writes `comparison.json` and `comparison.md` with both policies
side by side. Runs with the same seed, configuration, and deterministic policy
produce identical artifacts apart from timestamps and durations.

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

## Development

```bash
uv sync
uv run pytest -q
```

Live Ollama smoke tests are opt-in:

```bash
OLLAMA_LIVE=1 uv run pytest -q tests/test_live_smoke.py
```

## Limitations

- The search is a local tree search; it offers no global optimality guarantee.
- Expression size, operator set, and operand bank are bounded for tractability.
- The decision policy requires a compatible Ollama build and model; the offline
  policy is the reproducible fallback.
- The built-in suite is small and synthetic; results on real datasets depend on
  feature engineering and search budget.
- This is research code. APIs may change before a stable release.

## Contributing

Issues and pull requests are welcome. Please include a minimal reproduction,
the Python version, and whether the run used `offline`, `uniform`, or
`decision`.

## License

MIT. See [LICENSE](LICENSE).
