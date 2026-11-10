import numpy as np
import sympy as sp

from symbolic_regression.expression import (
    BinaryOp,
    Constant,
    UnaryOp,
    Variable,
    evaluate,
)
from symbolic_regression.simplify import simplify_expression, to_sympy


def test_to_sympy_matches_evaluation() -> None:
    expr = UnaryOp(
        "sin",
        BinaryOp(
            "add",
            BinaryOp("mul", Constant(2.0), Variable("x")),
            BinaryOp("div", Constant(0.5), Variable("y")),
        ),
    )
    sympy_expr = to_sympy(expr)
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(0.5, 2.0, 5)
    func = sp.lambdify(
        (sp.Symbol("x", real=True), sp.Symbol("y", real=True)), sympy_expr, "numpy"
    )
    np.testing.assert_allclose(func(x, y), np.sin(2.0 * x + 0.5 / y))
    np.testing.assert_allclose(
        evaluate(expr, {"x": x, "y": y}), np.sin(2.0 * x + 0.5 / y)
    )


def test_multiplication_simplifies_to_power() -> None:
    expr = BinaryOp("mul", Variable("x"), Variable("x"))
    assert simplify_expression(expr) == "x**2"


def test_decimal_constants_become_exact_rationals() -> None:
    expr = BinaryOp("add", Variable("x"), Constant(3.5))
    assert simplify_expression(expr) == "x + 7/2"


def test_redundant_terms_are_collected() -> None:
    expr = BinaryOp(
        "add",
        BinaryOp("mul", Variable("x"), Variable("x")),
        BinaryOp("mul", Constant(2.0), Variable("x")),
    )
    assert simplify_expression(expr) == "x*(x + 2)"


def test_real_variables_keep_absolute_value() -> None:
    expr = UnaryOp("abs", Variable("x"))
    assert simplify_expression(expr) == "Abs(x)"
    assert to_sympy(expr).has(sp.Abs)
