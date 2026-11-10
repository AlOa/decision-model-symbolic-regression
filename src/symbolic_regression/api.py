"""Public, programmatic API for symbolic regression.

The :func:`fit` function is the main entry point. It accepts features and a
target, optionally restricts the operator set, runs the MCTS search, and returns
a :class:`FitResult` with the recovered expression and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .data import Dataset
from .decision.client import OllamaClient, OllamaError
from .decision.policy import (
    OfflineOperatorPolicy,
    OllamaOperatorPolicy,
    OperatorPolicy,
    UniformOperatorPolicy,
)
from .expression import (
    OPERATORS,
    BinaryOp,
    Constant,
    Expr,
    UnaryOp,
    Variable,
    evaluate,
    to_string,
)
from .metrics import Evaluation
from .result import SearchResult, StopReason
from .search.mcts import MCTS, MCTSConfig

__all__ = [
    "FitResult",
    "SymbolicRegressor",
    "available_operators",
    "fit",
]


def available_operators() -> tuple[str, ...]:
    """Return the names of all registered operators."""

    return tuple(OPERATORS)


def _normalize_name(value: object) -> str:
    name = str(value)
    if not name.strip():
        raise ValueError("Feature and target names must not be empty")
    return name


def _unique_names(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name in seen:
            raise ValueError(f"Duplicate feature name: {name!r}")
        seen.add(name)
        unique.append(name)
    return unique


def _is_frame(value: object) -> bool:
    return (
        not isinstance(value, Mapping)
        and hasattr(value, "columns")
        and hasattr(value, "__getitem__")
    )


def _column_values(container: Any, label: object) -> np.ndarray:
    column = container[label]
    to_numpy = getattr(column, "to_numpy", None)
    values = to_numpy() if callable(to_numpy) else np.asarray(column)
    return np.asarray(values, dtype=float)


def _named_columns(container: Any) -> dict[str, object]:
    """Map normalized column names to the container's native labels."""

    labels = list(container.columns)
    mapping: dict[str, object] = {}
    for label in labels:
        name = _normalize_name(label)
        if name in mapping:
            raise ValueError(f"Duplicate column name after string conversion: {name!r}")
        mapping[name] = label
    return mapping


def _select_names(
    available: Mapping[str, object], requested: Sequence[str] | None
) -> list[str]:
    if requested is None:
        selected = list(available)
    else:
        if isinstance(requested, str):
            requested = [requested]
        selected = [_normalize_name(name) for name in requested]
        selected = _unique_names(selected)
        missing = [name for name in selected if name not in available]
        if missing:
            raise ValueError(
                f"Feature column(s) not found: {', '.join(missing)}. "
                f"Available columns: {', '.join(available)}"
            )
    if not selected:
        raise ValueError("At least one feature column is required")
    return selected


def _coerce_dataset(
    features: Any,
    target: Any,
    feature_names: Sequence[str] | None,
) -> Dataset:
    if target is None:
        raise ValueError("A target array or target column name is required")

    if _is_frame(features):
        columns = _named_columns(features)
        if isinstance(target, str):
            if target not in columns:
                raise ValueError(
                    f"Target column {target!r} not found. "
                    f"Available columns: {', '.join(columns)}"
                )
            target_values = _column_values(features, columns.pop(target))
        else:
            target_values = target
        selected = _select_names(columns, feature_names)
        data = {
            name: _column_values(features, columns[name]) for name in selected
        }
        return Dataset.from_arrays(data, target_values)

    if isinstance(features, Mapping):
        columns = {
            _normalize_name(key): key for key in features
        }
        if len(columns) != len(features):
            raise ValueError("Duplicate feature name after string conversion")
        if isinstance(target, str):
            if target not in columns:
                raise ValueError(
                    f"Target column {target!r} not found. "
                    f"Available columns: {', '.join(columns)}"
                )
            target_values = features[columns.pop(target)]
        else:
            target_values = target
        selected = _select_names(columns, feature_names)
        data = {
            name: np.asarray(features[columns[name]], dtype=float)
            for name in selected
        }
        return Dataset.from_arrays(data, target_values)

    if isinstance(target, str):
        raise ValueError(
            "target must be an array when features is a bare array; "
            "use a mapping or a data frame to pass a target column name"
        )
    array = np.asarray(features, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("features must be one- or two-dimensional")
    n_features = int(array.shape[1])
    if feature_names is None:
        names = ["x"] if n_features == 1 else [f"x{index}" for index in range(n_features)]
    else:
        names = _unique_names([_normalize_name(name) for name in feature_names])
        if len(names) != n_features:
            raise ValueError(
                f"feature_names has {len(names)} names but features have {n_features} columns"
            )
    data = {names[index]: array[:, index] for index in range(n_features)}
    return Dataset.from_arrays(data, target)


def _resolve_operators(operators: Sequence[str] | str | None) -> tuple[str, ...]:
    if operators is None:
        return tuple(OPERATORS)
    if isinstance(operators, str):
        names = [item.strip() for item in operators.split(",") if item.strip()]
    else:
        names = [_normalize_name(name) for name in operators]
    names = _unique_names(names)
    if not names:
        raise ValueError("operators must contain at least one operator")
    unknown = [name for name in names if name not in OPERATORS]
    if unknown:
        raise ValueError(
            f"Unknown operator(s): {', '.join(unknown)}. "
            f"Available operators: {', '.join(available_operators())}"
        )
    return tuple(names)


def _initial_expression(value: Expr | str | None, dataset: Dataset) -> Expr:
    if value is None:
        return Variable(dataset.feature_names[0])
    if isinstance(value, str):
        if value not in dataset.features:
            raise ValueError(
                f"Initial expression variable {value!r} is not a feature. "
                f"Available features: {', '.join(dataset.feature_names)}"
            )
        return Variable(value)
    if isinstance(value, (Variable, Constant, UnaryOp, BinaryOp)):
        return value
    raise TypeError("initial_expression must be an expression node or a feature name")


def _build_policy(
    policy: str | OperatorPolicy,
    operators: tuple[str, ...],
    *,
    seed: int,
    model: str,
    base_url: str,
    request_timeout: float,
    request_retries: int,
    client: OllamaClient | None,
) -> tuple[OperatorPolicy, OllamaClient | None]:
    if not isinstance(policy, str):
        if not callable(getattr(policy, "probabilities", None)):
            raise TypeError(
                "policy must be 'decision', 'uniform', 'offline', or an "
                "OperatorPolicy instance"
            )
        return policy, None
    kind = policy.lower()
    if kind == "decision":
        if len(operators) < 2:
            raise ValueError("The decision policy needs at least two operators")
        owned_client: OllamaClient | None = None
        active_client = client
        if active_client is None:
            active_client = OllamaClient(
                base_url=base_url,
                timeout=request_timeout,
                retries=request_retries,
            )
            owned_client = active_client
        try:
            active_client.readiness(model)
        except OllamaError:
            if owned_client is not None:
                owned_client.close()
            raise
        return (
            OllamaOperatorPolicy(
                client=active_client, model=model, operators=operators
            ),
            owned_client,
        )
    if kind == "uniform":
        return UniformOperatorPolicy(operators=operators), None
    if kind == "offline":
        return OfflineOperatorPolicy(seed=seed, operators=operators), None
    raise ValueError(
        f"Unknown policy {policy!r}; expected decision, uniform, offline, "
        "or an OperatorPolicy instance"
    )


@dataclass(slots=True)
class FitResult:
    """Recovered expression and diagnostics returned by :func:`fit`."""

    expression: Expr | None
    expression_str: str | None
    simplified_expression: str | None
    evaluation: Evaluation | None
    reward: float
    steps: int
    stop_reason: StopReason
    iterations: int
    decision_calls: int
    fallback_calls: int
    elapsed_seconds: float
    feature_names: tuple[str, ...]
    n_rows: int
    operators: tuple[str, ...]
    policy: str
    history: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.evaluation is not None and self.evaluation.valid

    @property
    def mse(self) -> float | None:
        return self.evaluation.mse if self.valid and self.evaluation else None

    @property
    def nmse(self) -> float | None:
        return self.evaluation.nmse if self.valid and self.evaluation else None

    @property
    def r2(self) -> float | None:
        return self.evaluation.r2 if self.valid and self.evaluation else None

    def predict(self, features: Any) -> np.ndarray:
        """Evaluate the recovered expression on new feature values."""

        if self.expression is None:
            raise ValueError("No feasible expression was found")
        mapping = self._prediction_features(features)
        return evaluate(self.expression, mapping)

    def _prediction_features(self, features: Any) -> dict[str, np.ndarray]:
        if _is_frame(features):
            columns = _named_columns(features)
            missing = [name for name in self.feature_names if name not in columns]
            if missing:
                raise ValueError(f"Missing feature column(s): {', '.join(missing)}")
            return {
                name: _column_values(features, columns[name])
                for name in self.feature_names
            }
        if isinstance(features, Mapping):
            columns = {_normalize_name(key): key for key in features}
            missing = [name for name in self.feature_names if name not in columns]
            if missing:
                raise ValueError(f"Missing feature column(s): {', '.join(missing)}")
            return {
                name: np.asarray(features[columns[name]], dtype=float)
                for name in self.feature_names
            }
        array = np.asarray(features, dtype=float)
        if array.ndim == 1:
            if len(self.feature_names) != 1:
                raise ValueError(
                    f"Expected {len(self.feature_names)} feature columns, got one"
                )
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("features must be one- or two-dimensional")
        if int(array.shape[1]) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} feature columns, "
                f"got {int(array.shape[1])}"
            )
        return {
            name: array[:, index]
            for index, name in enumerate(self.feature_names)
        }

    def to_dict(self, include_trace: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "expression": self.expression_str,
            "simplified_expression": self.simplified_expression,
            "metrics": self.evaluation.to_dict() if self.evaluation is not None else None,
            "reward": float(self.reward),
            "steps": int(self.steps),
            "stop_reason": self.stop_reason.value,
            "iterations": int(self.iterations),
            "decision_calls": int(self.decision_calls),
            "fallback_calls": int(self.fallback_calls),
            "elapsed_seconds": float(self.elapsed_seconds),
            "policy": self.policy,
            "operators": list(self.operators),
            "feature_names": list(self.feature_names),
            "n_rows": int(self.n_rows),
        }
        if include_trace:
            payload["history"] = list(self.history)
            payload["decisions"] = list(self.decisions)
        return payload

    def summary(self) -> str:
        lines = [
            f"Expression: {self.expression_str or 'none'}",
        ]
        if self.simplified_expression and self.simplified_expression != self.expression_str:
            lines.append(f"Simplified expression: {self.simplified_expression}")
        if self.valid:
            lines.extend(
                [
                    f"MSE: {self.mse:.6g}",
                    f"NMSE: {self.nmse:.6g}",
                    f"R-squared: {self.r2:.6g}",
                ]
            )
        else:
            lines.append("Metrics: no feasible expression")
        lines.extend(
            [
                f"Stop reason: {self.stop_reason.value}",
                f"Steps: {self.steps}",
                f"Iterations: {self.iterations}",
                f"Decision calls: {self.decision_calls} (fallbacks: {self.fallback_calls})",
                f"Policy: {self.policy}",
                f"Operators: {', '.join(self.operators)}",
                f"Elapsed: {self.elapsed_seconds:.3f}s",
            ]
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def fit(
    features: Any,
    target: Any = None,
    *,
    feature_names: Sequence[str] | None = None,
    operators: Sequence[str] | str | None = None,
    policy: str | OperatorPolicy = "offline",
    model: str = "d3:4b",
    base_url: str = "http://localhost:11434",
    request_timeout: float = 30.0,
    request_retries: int = 1,
    client: OllamaClient | None = None,
    seed: int = 0,
    iterations: int = 50,
    max_nodes: int = 500,
    max_depth: int = 6,
    rollout_depth: int = 3,
    c_puct: float = 1.4,
    top_applications: int = 2,
    max_operands: int = 256,
    fit_threshold: float = 0.995,
    max_steps: int = 12,
    time_limit_seconds: float = 300.0,
    greedy: bool = False,
    context_rows: int = 24,
    residual_values: int = 16,
    initial_expression: Expr | str | None = None,
) -> FitResult:
    """Search for a symbolic expression that fits ``target``.

    Parameters
    ----------
    features:
        Feature data. Accepted forms are a mapping of name to 1-D values, a
        two-dimensional array (optionally with ``feature_names``), or a data
        frame exposing ``columns`` and column indexing. A data frame may include
        the target column when ``target`` is its name.
    target:
        Target values, or the name of the target column when ``features`` is a
        mapping or a data frame.
    operators:
        Operator names allowed in expressions. ``None`` uses every registered
        operator; a comma-separated string is also accepted.
    policy:
        ``"offline"`` (default, deterministic, no server), ``"uniform"``,
        ``"decision"`` (requires Ollama), or a custom policy instance.
    initial_expression:
        Starting expression node or feature name. Defaults to the first feature.

    All remaining keyword arguments control the MCTS search and the decision
    context. They mirror ``MCTSConfig`` and are optional.
    """

    dataset = _coerce_dataset(features, target, feature_names)
    selected_operators = _resolve_operators(operators)
    policy_object, owned_client = _build_policy(
        policy,
        selected_operators,
        seed=seed,
        model=model,
        base_url=base_url,
        request_timeout=request_timeout,
        request_retries=request_retries,
        client=client,
    )
    config = MCTSConfig(
        iterations=iterations,
        max_nodes=max_nodes,
        max_depth=max_depth,
        rollout_depth=rollout_depth,
        c_puct=c_puct,
        top_applications=top_applications,
        max_operands=max_operands,
        fit_threshold=fit_threshold,
        max_steps=max_steps,
        time_limit_seconds=time_limit_seconds,
        seed=seed,
        greedy=greedy,
        operators=selected_operators,
        context_rows=context_rows,
        residual_values=residual_values,
    )
    try:
        search = MCTS(
            initial_expression=_initial_expression(initial_expression, dataset),
            dataset=dataset,
            policy=policy_object,
            config=config,
        )
        result = search.run()
    finally:
        if owned_client is not None:
            owned_client.close()
    return _make_fit_result(
        result,
        dataset=dataset,
        policy_name=getattr(policy_object, "name", type(policy_object).__name__),
        operators=selected_operators,
    )


def _make_fit_result(
    result: SearchResult,
    *,
    dataset: Dataset,
    policy_name: str,
    operators: tuple[str, ...],
) -> FitResult:
    return FitResult(
        expression=result.best_expression,
        expression_str=(
            to_string(result.best_expression)
            if result.best_expression is not None
            else None
        ),
        simplified_expression=result.simplified_expression,
        evaluation=result.best_evaluation,
        reward=result.best_reward,
        steps=result.steps,
        stop_reason=result.stop_reason,
        iterations=result.iterations,
        decision_calls=result.decision_calls,
        fallback_calls=result.fallback_calls,
        elapsed_seconds=result.elapsed_seconds,
        feature_names=tuple(dataset.feature_names),
        n_rows=dataset.n_rows,
        operators=operators,
        policy=policy_name,
        history=list(result.history),
        decisions=list(result.decision_records),
    )


class SymbolicRegressor:
    """Small stateful wrapper around :func:`fit`.

    Search options passed to the constructor are reused by :meth:`fit`; values
    passed to :meth:`fit` override them.
    """

    def __init__(
        self,
        *,
        operators: Sequence[str] | str | None = None,
        policy: str | OperatorPolicy = "offline",
        **fit_options: Any,
    ) -> None:
        self.operators = operators
        self.policy = policy
        self.fit_options = dict(fit_options)
        self.result_: FitResult | None = None

    def fit(
        self,
        features: Any,
        target: Any = None,
        *,
        feature_names: Sequence[str] | None = None,
        operators: Sequence[str] | str | None = None,
        policy: str | OperatorPolicy | None = None,
        **options: Any,
    ) -> "SymbolicRegressor":
        merged = {**self.fit_options, **options}
        stored_feature_names = merged.pop("feature_names", None)
        selected_feature_names = (
            feature_names if feature_names is not None else stored_feature_names
        )
        self.result_ = fit(
            features,
            target,
            feature_names=selected_feature_names,
            operators=operators if operators is not None else self.operators,
            policy=policy if policy is not None else self.policy,
            **merged,
        )
        return self

    def _require_result(self) -> FitResult:
        if self.result_ is None:
            raise RuntimeError("Call fit() before accessing the result")
        return self.result_

    @property
    def expression_(self) -> Expr | None:
        return self._require_result().expression

    @property
    def expression_str_(self) -> str | None:
        return self._require_result().expression_str

    @property
    def simplified_expression_(self) -> str | None:
        return self._require_result().simplified_expression

    @property
    def metrics_(self) -> dict[str, Any] | None:
        result = self._require_result()
        return result.evaluation.to_dict() if result.evaluation is not None else None

    @property
    def diagnostics_(self) -> dict[str, Any]:
        return self._require_result().to_dict()

    def predict(self, features: Any) -> np.ndarray:
        return self._require_result().predict(features)

    def to_dict(self, include_trace: bool = True) -> dict[str, Any]:
        return self._require_result().to_dict(include_trace=include_trace)

    def summary(self) -> str:
        return self._require_result().summary()
