"""Deterministic operand bank used to apply binary operators."""

from __future__ import annotations

from typing import Sequence

from ..expression import (
    BinaryOp,
    Constant,
    Expr,
    UnaryOp,
    Variable,
    node_count,
    to_string,
)

DEFAULT_CONSTANTS: tuple[float, ...] = (-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 3.5)
DEFAULT_UNARY = ("sin", "cos", "exp", "log", "sqrt", "abs")


def build_operand_bank(
    variables: Sequence[str],
    constants: Sequence[float] = DEFAULT_CONSTANTS,
    unary_names: Sequence[str] = DEFAULT_UNARY,
    max_operands: int = 256,
) -> list[Expr]:
    """Return a deterministic, deduplicated list of candidate operands.

    The bank contains variables, constants, unary functions of variables, and a
    compact set of compound terms such as ``2*x``, ``x*x``, ``sin(3*x)`` and
    ``exp(x/-5)``. It is deliberately finite so binary applications stay
    tractable.
    """

    candidates: list[Expr] = []
    for name in variables:
        candidates.append(Variable(name))
    for value in constants:
        candidates.append(Constant(float(value)))

    compounds: list[Expr] = []
    for name in variables:
        variable = Variable(name)
        for value in (2.0, 3.0, 0.5, -1.0, 5.0):
            compounds.append(BinaryOp("mul", Constant(value), variable))
        for value in (2.0, 5.0, -5.0):
            compounds.append(BinaryOp("div", variable, Constant(value)))
        compounds.append(BinaryOp("mul", variable, variable))
        for value in (1.0, -1.0, 0.5):
            compounds.append(BinaryOp("add", variable, Constant(value)))
            compounds.append(BinaryOp("sub", variable, Constant(value)))
    if len(variables) > 1:
        for index, left_name in enumerate(variables):
            for right_name in variables[index + 1 :]:
                left, right = Variable(left_name), Variable(right_name)
                compounds.append(BinaryOp("mul", left, right))
                compounds.append(BinaryOp("add", left, right))
                compounds.append(BinaryOp("sub", left, right))

    candidates.extend(compounds)
    for unary in unary_names:
        for compound in compounds:
            candidates.append(UnaryOp(unary, compound))

    seen: set[str] = set()
    unique: list[Expr] = []
    for candidate in candidates:
        label = to_string(candidate)
        if label in seen:
            continue
        seen.add(label)
        unique.append(candidate)
    unique.sort(key=lambda expr: (node_count(expr), to_string(expr)))
    if max_operands is not None and max_operands >= 0:
        unique = unique[:max_operands]
    return unique
