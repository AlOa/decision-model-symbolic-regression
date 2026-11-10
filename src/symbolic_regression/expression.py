"""Immutable expression trees and the operator registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


class ExpressionError(ValueError):
    """Base error for expression construction and evaluation."""


class UnknownOperatorError(ExpressionError):
    """Raised when an operator name is not registered."""


class OperatorArityError(ExpressionError):
    """Raised when an operator is applied to the wrong number of operands."""


class InvalidEvaluation(ExpressionError):
    """Raised when an expression cannot be evaluated on the supplied data."""


@dataclass(frozen=True, slots=True)
class Variable:
    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ExpressionError("Variable name must not be empty")


@dataclass(frozen=True, slots=True)
class Constant:
    value: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ExpressionError("Constant must be finite")


@dataclass(frozen=True, slots=True)
class UnaryOp:
    name: str
    operand: "Expr"

    def __post_init__(self) -> None:
        if self.name not in UNARY_OPERATORS:
            raise UnknownOperatorError(f"Unknown unary operator: {self.name}")


@dataclass(frozen=True, slots=True)
class BinaryOp:
    name: str
    left: "Expr"
    right: "Expr"

    def __post_init__(self) -> None:
        if self.name not in BINARY_OPERATORS:
            raise UnknownOperatorError(f"Unknown binary operator: {self.name}")


Expr = Variable | Constant | UnaryOp | BinaryOp


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    name: str
    arity: int
    description: str
    function: Callable[..., np.ndarray]
    guard: Callable[..., np.ndarray] | None = None


def _guard_div(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(right) > 1e-12


def _guard_log(value: np.ndarray) -> np.ndarray:
    return value > 0


def _guard_sqrt(value: np.ndarray) -> np.ndarray:
    return value >= 0


def _guard_exp(value: np.ndarray) -> np.ndarray:
    return value < 700.0


BINARY_OPERATORS: dict[str, OperatorSpec] = {
    "add": OperatorSpec("add", 2, "addition", lambda a, b: a + b),
    "sub": OperatorSpec("sub", 2, "subtraction", lambda a, b: a - b),
    "mul": OperatorSpec("mul", 2, "multiplication", lambda a, b: a * b),
    "div": OperatorSpec("div", 2, "division", lambda a, b: a / b, _guard_div),
}

UNARY_OPERATORS: dict[str, OperatorSpec] = {
    "sin": OperatorSpec("sin", 1, "sine", np.sin),
    "cos": OperatorSpec("cos", 1, "cosine", np.cos),
    "exp": OperatorSpec("exp", 1, "exponential", np.exp, _guard_exp),
    "log": OperatorSpec("log", 1, "natural logarithm", np.log, _guard_log),
    "sqrt": OperatorSpec("sqrt", 1, "square root", np.sqrt, _guard_sqrt),
    "abs": OperatorSpec("abs", 1, "absolute value", np.abs),
}

OPERATORS: dict[str, OperatorSpec] = {**BINARY_OPERATORS, **UNARY_OPERATORS}

BINARY_SYMBOLS: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


def apply_operator(name: str, operands: Sequence[Expr]) -> Expr:
    """Return a new expression with ``name`` applied to ``operands``.

    The current expressions are never mutated because every node is immutable.
    """

    spec = OPERATORS.get(name)
    if spec is None:
        raise UnknownOperatorError(f"Unknown operator: {name}")
    if len(operands) != spec.arity:
        raise OperatorArityError(
            f"Operator {name!r} expects {spec.arity} operand(s), got {len(operands)}"
        )
    if spec.arity == 1:
        return UnaryOp(name, operands[0])
    return BinaryOp(name, operands[0], operands[1])


def _n_rows(features: Mapping[str, np.ndarray], n_rows: int | None) -> int:
    if n_rows is not None:
        return n_rows
    for values in features.values():
        array = np.asarray(values)
        if array.ndim != 1:
            raise InvalidEvaluation("Feature arrays must be one-dimensional")
        return int(array.size)
    raise InvalidEvaluation("Cannot infer row count from an empty feature set")


def evaluate(expr: Expr, features: Mapping[str, np.ndarray], n_rows: int | None = None) -> np.ndarray:
    """Evaluate ``expr`` on the supplied feature columns.

    Raises :class:`InvalidEvaluation` for missing variables, domain violations,
    shape mismatches, or non-finite results.
    """

    rows = _n_rows(features, n_rows)

    def visit(node: Expr) -> np.ndarray:
        if isinstance(node, Variable):
            if node.name not in features:
                raise InvalidEvaluation(f"Unknown variable: {node.name}")
            values = np.asarray(features[node.name], dtype=float)
            if values.ndim != 1 or values.size != rows:
                raise InvalidEvaluation(f"Variable {node.name!r} has an unexpected shape")
            return values
        if isinstance(node, Constant):
            return np.full(rows, float(node.value), dtype=float)
        if isinstance(node, UnaryOp):
            operand = visit(node.operand)
            spec = UNARY_OPERATORS[node.name]
            if spec.guard is not None and not np.all(spec.guard(operand)):
                raise InvalidEvaluation(f"Operator {node.name!r} received an invalid domain")
            with np.errstate(all="ignore"):
                result = np.asarray(spec.function(operand), dtype=float)
        elif isinstance(node, BinaryOp):
            left = visit(node.left)
            right = visit(node.right)
            spec = BINARY_OPERATORS[node.name]
            if spec.guard is not None and not np.all(spec.guard(left, right)):
                raise InvalidEvaluation(f"Operator {node.name!r} received an invalid domain")
            with np.errstate(all="ignore"):
                result = np.asarray(spec.function(left, right), dtype=float)
        else:  # pragma: no cover - defensive
            raise InvalidEvaluation(f"Unsupported expression node: {type(node)!r}")
        if result.ndim != 1 or result.size != rows:
            raise InvalidEvaluation("Expression produced an unexpected shape")
        if not np.all(np.isfinite(result)):
            raise InvalidEvaluation("Expression produced a non-finite value")
        return result

    return visit(expr)


def node_count(expr: Expr) -> int:
    if isinstance(expr, (Variable, Constant)):
        return 1
    if isinstance(expr, UnaryOp):
        return 1 + node_count(expr.operand)
    if isinstance(expr, BinaryOp):
        return 1 + node_count(expr.left) + node_count(expr.right)
    raise ExpressionError(f"Unsupported expression node: {type(expr)!r}")


def variables(expr: Expr) -> tuple[str, ...]:
    found: list[str] = []

    def visit(node: Expr) -> None:
        if isinstance(node, Variable):
            if node.name not in found:
                found.append(node.name)
        elif isinstance(node, UnaryOp):
            visit(node.operand)
        elif isinstance(node, BinaryOp):
            visit(node.left)
            visit(node.right)

    visit(expr)
    return tuple(found)


def subexpressions(expr: Expr) -> tuple[Expr, ...]:
    found: list[Expr] = []

    def visit(node: Expr) -> None:
        found.append(node)
        if isinstance(node, UnaryOp):
            visit(node.operand)
        elif isinstance(node, BinaryOp):
            visit(node.left)
            visit(node.right)

    visit(expr)
    return tuple(found)


def _format_constant(value: float) -> str:
    return format(float(value), ".6g")


def to_string(expr: Expr) -> str:
    if isinstance(expr, Variable):
        return expr.name
    if isinstance(expr, Constant):
        return _format_constant(expr.value)
    if isinstance(expr, UnaryOp):
        return f"{expr.name}({to_string(expr.operand)})"
    if isinstance(expr, BinaryOp):
        operator = BINARY_SYMBOLS[expr.name]
        return f"({to_string(expr.left)} {operator} {to_string(expr.right)})"
    raise ExpressionError(f"Unsupported expression node: {type(expr)!r}")


def iter_nodes(expr: Expr) -> Iterable[Expr]:
    yield expr
    if isinstance(expr, UnaryOp):
        yield from iter_nodes(expr.operand)
    elif isinstance(expr, BinaryOp):
        yield from iter_nodes(expr.left)
        yield from iter_nodes(expr.right)
