"""Best-effort symbolic simplification of recovered expressions with SymPy."""

from __future__ import annotations

import sympy as sp

from .expression import BinaryOp, Constant, Expr, UnaryOp, Variable

_UNARY_SYMPY = {
    "sin": sp.sin,
    "cos": sp.cos,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}

_BINARY_SYMPY = {
    "add": lambda left, right: left + right,
    "sub": lambda left, right: left - right,
    "mul": lambda left, right: left * right,
    "div": lambda left, right: left / right,
}


def to_sympy(expr: Expr) -> sp.Expr:
    """Convert an expression tree into an equivalent SymPy expression.

    Variables are declared real and constants are turned into exact rationals,
    so simple decimal constants such as ``0.5`` are rendered as ``1/2``.
    """

    if isinstance(expr, Variable):
        return sp.Symbol(expr.name, real=True)
    if isinstance(expr, Constant):
        return sp.nsimplify(float(expr.value), rational=True)
    if isinstance(expr, UnaryOp):
        return _UNARY_SYMPY[expr.name](to_sympy(expr.operand))
    if isinstance(expr, BinaryOp):
        return _BINARY_SYMPY[expr.name](to_sympy(expr.left), to_sympy(expr.right))
    raise TypeError(f"Unsupported expression node: {type(expr)!r}")


def simplify_expression(expr: Expr) -> str | None:
    """Return a simplified SymPy rendering of ``expr``, or ``None`` on failure.

    Simplification is informational: the search keeps the original expression
    and its measured metrics, so a failure here never invalidates a run.
    """

    try:
        simplified = sp.simplify(to_sympy(expr))
    except (sp.SympifyError, TypeError, ValueError, ArithmeticError, RecursionError):
        return None
    if simplified.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        return None
    return sp.sstr(simplified)
