"""Compact, deterministic decision-context serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data import Dataset
from .expression import Expr, to_string
from .metrics import evaluate_expression


CONTEXT_SCHEMA_VERSION = 1


@dataclass(slots=True)
class DecisionContext:
    step: int
    expression: str
    features: dict[str, list[float]]
    target: list[float]
    residual: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "step": self.step,
            "expression": self.expression,
            "features": {name: list(values) for name, values in self.features.items()},
            "target": list(self.target),
            "residual": dict(self.residual),
            "history": [dict(record) for record in self.history],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False)


def _round(value: float, digits: int | None) -> float:
    numeric = float(value)
    if digits is None:
        return numeric
    return round(numeric, digits)


def _sample_indices(n_rows: int, count: int, seed: int, step: int) -> np.ndarray:
    size = min(n_rows, max(1, count))
    if size >= n_rows:
        return np.arange(n_rows, dtype=int)
    rng = np.random.default_rng(seed + step)
    return np.sort(rng.choice(n_rows, size=size, replace=False))


def build_context(
    expr: Expr,
    dataset: Dataset,
    step: int,
    history: list[dict[str, Any]] | None = None,
    context_rows: int = 24,
    residual_values: int = 16,
    seed: int = 0,
    rounding: int | None = 4,
) -> DecisionContext:
    """Build the compact state sent to a decision model."""

    indices = _sample_indices(dataset.n_rows, context_rows, seed, step)
    features = {
        name: [_round(values[index], rounding) for index in indices]
        for name, values in dataset.features.items()
    }
    target = [_round(dataset.target[index], rounding) for index in indices]
    evaluation = evaluate_expression(expr, dataset)
    if not evaluation.valid or evaluation.residual is None:
        residual_data: dict[str, Any] = {
            "available": False,
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "values": [],
        }
    else:
        values = evaluation.residual
        residual_indices = indices[: max(1, residual_values)]
        residual_data = {
            "available": True,
            "count": int(values.size),
            "min": _round(float(np.min(values)), rounding),
            "max": _round(float(np.max(values)), rounding),
            "mean": _round(float(np.mean(values)), rounding),
            "std": _round(float(np.std(values)), rounding),
            "values": [_round(values[index], rounding) for index in residual_indices],
        }
    return DecisionContext(
        step=step,
        expression=to_string(expr),
        features=features,
        target=target,
        residual=residual_data,
        history=list(history or []),
    )
