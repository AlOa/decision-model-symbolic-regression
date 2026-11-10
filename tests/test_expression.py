import numpy as np
import pytest

from symbolic_regression.expression import (
    BinaryOp,
    Constant,
    InvalidEvaluation,
    OperatorArityError,
    UnaryOp,
    Variable,
    apply_operator,
    evaluate,
    node_count,
    to_string,
    variables,
)


def test_unary_and_binary_construction_and_evaluation() -> None:
    expr = UnaryOp("sin", BinaryOp("mul", Variable("x"), Constant(2.0)))
    x = np.array([0.0, np.pi / 4, np.pi / 2])
    result = evaluate(expr, {"x": x})
    np.testing.assert_allclose(result, np.sin(2 * x))
    assert node_count(expr) == 4
    assert variables(expr) == ("x",)


def test_invalid_arity_is_rejected() -> None:
    with pytest.raises(OperatorArityError):
        apply_operator("sin", [Variable("x"), Variable("x")])
    with pytest.raises(OperatorArityError):
        apply_operator("add", [Variable("x")])


def test_formatting_has_explicit_parentheses() -> None:
    expr = BinaryOp("add", BinaryOp("mul", Variable("x"), Constant(2)), Variable("y"))
    assert to_string(expr) == "((x * 2) + y)"


def test_division_by_zero_is_invalid() -> None:
    expr = BinaryOp("div", Variable("x"), Constant(0.0))
    with pytest.raises(InvalidEvaluation):
        evaluate(expr, {"x": np.array([1.0, 2.0])})


def test_log_and_sqrt_domains_are_guarded() -> None:
    with pytest.raises(InvalidEvaluation):
        evaluate(UnaryOp("log", Variable("x")), {"x": np.array([0.0, 1.0])})
    with pytest.raises(InvalidEvaluation):
        evaluate(UnaryOp("sqrt", Variable("x")), {"x": np.array([-1.0, 1.0])})


def test_constant_expression_uses_explicit_row_count() -> None:
    result = evaluate(Constant(3.5), {}, n_rows=3)
    np.testing.assert_allclose(result, np.array([3.5, 3.5, 3.5]))


def test_suite_dataset_variables_evaluate(suite_dataset) -> None:
    name, label, dataset = suite_dataset
    assert name and label
    for variable in dataset.feature_names:
        result = evaluate(Variable(variable), dataset.features, n_rows=dataset.n_rows)
        assert result.shape == (dataset.n_rows,)
        assert np.all(np.isfinite(result))
