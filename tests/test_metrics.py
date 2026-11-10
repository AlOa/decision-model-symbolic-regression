import numpy as np
import pytest

from symbolic_regression.data import Dataset
from symbolic_regression.expression import BinaryOp, Constant, Variable
from symbolic_regression.metrics import evaluate_expression, reward


@pytest.fixture
def dataset() -> Dataset:
    x = np.linspace(-2, 2, 9)
    return Dataset.from_arrays({"x": x}, x**2)


def test_exact_expression_has_zero_residual_and_perfect_fit(dataset) -> None:
    evaluation = evaluate_expression(BinaryOp("mul", Variable("x"), Variable("x")), dataset)
    assert evaluation.valid
    assert evaluation.mse == pytest.approx(0.0)
    assert evaluation.nmse == pytest.approx(0.0)
    assert evaluation.r2 == pytest.approx(1.0)
    assert reward(evaluation) == pytest.approx(1.0)
    np.testing.assert_allclose(evaluation.residual, np.zeros(dataset.n_rows))


def test_non_finite_prediction_is_invalid(dataset) -> None:
    evaluation = evaluate_expression(BinaryOp("div", Constant(1.0), Constant(0.0)), dataset)
    assert not evaluation.valid
    assert evaluation.error
    assert reward(evaluation) == 0.0


def test_constant_target_is_handled() -> None:
    dataset = Dataset.from_arrays({"x": [0.0, 1.0, 2.0]}, [3.5, 3.5, 3.5])
    evaluation = evaluate_expression(Constant(3.5), dataset)
    assert evaluation.valid
    assert evaluation.nmse == pytest.approx(0.0)
    assert evaluation.r2 == pytest.approx(1.0)


def test_suite_dataset_metrics_are_valid(suite_dataset) -> None:
    name, label, dataset = suite_dataset
    assert name and label
    evaluation = evaluate_expression(Constant(0.0), dataset)
    assert evaluation.valid
    assert evaluation.mse is not None
    assert evaluation.nmse is not None
    assert 0.0 <= reward(evaluation) <= 1.0
