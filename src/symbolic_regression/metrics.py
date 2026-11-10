"""Prediction, residual, and fit metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import Dataset
from .expression import Expr, ExpressionError, evaluate


@dataclass(slots=True)
class Evaluation:
    predictions: np.ndarray | None
    residual: np.ndarray | None
    mse: float | None
    nmse: float | None
    r2: float | None
    valid: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "error": self.error,
            "mse": _finite_or_none(self.mse),
            "nmse": _finite_or_none(self.nmse),
            "r2": _finite_or_none(self.r2),
        }


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(value) else None


def evaluate_expression(expr: Expr, dataset: Dataset) -> Evaluation:
    """Evaluate an expression and compute fit metrics on the dataset."""

    try:
        predictions = evaluate(expr, dataset.features, n_rows=dataset.n_rows)
    except (ExpressionError, ValueError, FloatingPointError) as exc:
        return Evaluation(
            predictions=None,
            residual=None,
            mse=None,
            nmse=None,
            r2=None,
            valid=False,
            error=str(exc),
        )
    residual = predictions - dataset.target
    mse = float(np.mean(residual**2))
    centered = dataset.target - float(np.mean(dataset.target))
    sst = float(np.sum(centered**2))
    sse = float(np.sum(residual**2))
    if sst > 0:
        nmse = sse / sst
        r2 = 1.0 - nmse
    elif sse == 0:
        nmse = 0.0
        r2 = 1.0
    else:
        nmse = float("inf")
        r2 = 0.0
    if not math.isfinite(mse):
        return Evaluation(
            predictions=None,
            residual=None,
            mse=None,
            nmse=None,
            r2=None,
            valid=False,
            error="fit metrics are not finite",
        )
    return Evaluation(
        predictions=predictions,
        residual=residual,
        mse=mse,
        nmse=nmse,
        r2=r2,
        valid=True,
    )


def reward(evaluation: Evaluation) -> float:
    """Map an evaluation to a reward in [0, 1]; invalid results get zero."""

    if not evaluation.valid or evaluation.nmse is None:
        return 0.0
    return float(1.0 / (1.0 + max(0.0, evaluation.nmse)))


def residual(expr: Expr, dataset: Dataset) -> np.ndarray | None:
    evaluation = evaluate_expression(expr, dataset)
    return None if evaluation.residual is None else evaluation.residual


def mse(expr: Expr, dataset: Dataset) -> float | None:
    return evaluate_expression(expr, dataset).mse


def nmse(expr: Expr, dataset: Dataset) -> float | None:
    return evaluate_expression(expr, dataset).nmse


def r2(expr: Expr, dataset: Dataset) -> float | None:
    return evaluate_expression(expr, dataset).r2
