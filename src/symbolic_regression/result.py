"""Stopping criteria and search/run result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .expression import Expr, to_string
from .metrics import Evaluation, reward
from .simplify import simplify_expression


class StopReason(str, Enum):
    FIT_THRESHOLD = "fit_threshold"
    MAX_STEPS = "max_steps"
    NO_VALID_OPERATOR = "no_valid_operator"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ERROR = "error"


@dataclass(slots=True)
class StopDecision:
    should_stop: bool
    reason: StopReason | None = None
    message: str = ""


def check_stopping(
    step: int,
    max_steps: int,
    evaluation: Evaluation | None,
    fit_threshold: float,
    has_valid_actions: bool = True,
) -> StopDecision:
    """Return whether construction should stop and why."""

    if evaluation is not None and evaluation.valid and evaluation.r2 is not None:
        if evaluation.r2 >= fit_threshold:
            return StopDecision(
                True,
                StopReason.FIT_THRESHOLD,
                f"R-squared {evaluation.r2:.6g} reached threshold {fit_threshold:.6g}",
            )
    if step >= max_steps:
        return StopDecision(
            True,
            StopReason.MAX_STEPS,
            f"maximum construction steps reached ({max_steps})",
        )
    if not has_valid_actions:
        return StopDecision(
            True,
            StopReason.NO_VALID_OPERATOR,
            "no valid operator can be applied",
        )
    return StopDecision(False)


@dataclass(slots=True)
class SearchResult:
    best_expression: Expr | None
    best_evaluation: Evaluation | None
    best_reward: float
    steps: int
    stop_reason: StopReason
    history: list[dict[str, Any]] = field(default_factory=list)
    decision_records: list[dict[str, Any]] = field(default_factory=list)
    decision_calls: int = 0
    fallback_calls: int = 0
    iterations: int = 0
    elapsed_seconds: float = 0.0
    simplified_expression: str | None = None

    def __post_init__(self) -> None:
        if self.simplified_expression is None and self.best_expression is not None:
            self.simplified_expression = simplify_expression(self.best_expression)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_expression": to_string(self.best_expression) if self.best_expression else None,
            "simplified_expression": self.simplified_expression,
            "best_metrics": self.best_evaluation.to_dict() if self.best_evaluation else None,
            "best_reward": float(self.best_reward),
            "steps": int(self.steps),
            "stop_reason": self.stop_reason.value,
            "iterations": int(self.iterations),
            "decision_calls": int(self.decision_calls),
            "fallback_calls": int(self.fallback_calls),
            "elapsed_seconds": float(self.elapsed_seconds),
        }


def reward_of(evaluation: Evaluation | None) -> float:
    return reward(evaluation) if evaluation is not None else 0.0
