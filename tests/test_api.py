import numpy as np
import pytest

from symbolic_regression import (
    SymbolicRegressor,
    available_operators,
    fit,
)
from symbolic_regression.expression import BinaryOp, to_string


class Frame:
    """Minimal data-frame lookalike used to test duck-typed input."""

    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        self._columns = columns

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._columns[name]


def small_fit(**overrides):
    x = np.linspace(-2.0, 2.0, 21)
    options = dict(
        operators=("mul",),
        iterations=5,
        max_nodes=50,
        max_depth=2,
        rollout_depth=0,
        max_steps=2,
        max_operands=32,
    )
    options.update(overrides)
    return fit({"x": x}, x**2, **options)


def test_fit_returns_expression_and_diagnostics() -> None:
    result = small_fit()
    assert result.valid
    assert result.expression is not None
    assert result.expression_str == "(x * x)"
    assert result.simplified_expression == "x**2"
    assert result.r2 == pytest.approx(1.0)
    assert result.stop_reason.value == "fit_threshold"
    assert result.policy == "offline"
    assert result.operators == ("mul",)
    assert result.feature_names == ("x",)
    assert result.n_rows == 21
    assert result.history
    assert result.decisions
    payload = result.to_dict()
    assert payload["expression"] == "(x * x)"
    assert payload["metrics"]["r2"] == pytest.approx(1.0)
    assert payload["history"]
    assert payload["decisions"]


def test_operator_set_is_enforced() -> None:
    result = small_fit()
    assert isinstance(result.expression, BinaryOp)
    assert result.expression.name == "mul"
    assert all(record["candidates"] == ["mul"] for record in result.decisions)
    assert all(record["expanded_operator"] == "mul" for record in result.history)
    assert to_string(result.expression) == "(x * x)"


def test_to_dict_can_omit_trace() -> None:
    payload = small_fit().to_dict(include_trace=False)
    assert "history" not in payload
    assert "decisions" not in payload


def test_fit_accepts_mapping_with_target_column() -> None:
    x = np.linspace(-2.0, 2.0, 21)
    result = fit(
        {"x": x, "y": x**2},
        "y",
        operators=("mul",),
        iterations=3,
        max_depth=2,
        rollout_depth=0,
        max_steps=2,
        max_operands=32,
    )
    assert result.feature_names == ("x",)
    assert result.simplified_expression == "x**2"


def test_fit_accepts_two_dimensional_array() -> None:
    x = np.linspace(-2.0, 2.0, 21)
    result = fit(
        x.reshape(-1, 1),
        x**2,
        feature_names=["t"],
        operators=("mul",),
        iterations=3,
        max_depth=2,
        rollout_depth=0,
        max_steps=2,
        max_operands=32,
    )
    assert result.feature_names == ("t",)
    assert result.simplified_expression == "t**2"


def test_fit_accepts_dataframe_lookalike() -> None:
    x = np.linspace(-2.0, 2.0, 21)
    frame = Frame({"x": x, "y": x**2})
    result = fit(
        frame,
        "y",
        operators=("mul",),
        iterations=3,
        max_depth=2,
        rollout_depth=0,
        max_steps=2,
        max_operands=32,
    )
    assert result.feature_names == ("x",)
    assert result.simplified_expression == "x**2"


def test_predict_reuses_the_recovered_expression() -> None:
    result = small_fit()
    np.testing.assert_allclose(result.predict({"x": [3.0]}), [9.0])
    np.testing.assert_allclose(result.predict([[3.0]]), [9.0])
    assert result.predict([3.0]).shape == (1,)


def test_regressor_wrapper() -> None:
    x = np.linspace(-2.0, 2.0, 21)
    model = SymbolicRegressor(
        operators=("mul",),
        iterations=3,
        max_depth=2,
        rollout_depth=0,
        max_steps=2,
        max_operands=32,
    )
    returned = model.fit(x.reshape(-1, 1), x**2)
    assert returned is model
    assert model.expression_ is not None
    assert model.expression_str_ is not None
    assert model.simplified_expression_ == "x**2"
    assert model.metrics_["r2"] == pytest.approx(1.0)
    assert model.diagnostics_["stop_reason"] == "fit_threshold"
    np.testing.assert_allclose(model.predict([[3.0]]), [9.0])


def test_regressor_requires_fit_before_result() -> None:
    model = SymbolicRegressor()
    with pytest.raises(RuntimeError):
        _ = model.expression_


def test_fit_accepts_custom_policy_instance() -> None:
    from symbolic_regression.decision.policy import UniformOperatorPolicy

    result = small_fit(policy=UniformOperatorPolicy(operators=("mul",)))
    assert result.policy == "uniform"
    assert result.simplified_expression == "x**2"


def test_available_operators_contains_registered_names() -> None:
    names = available_operators()
    assert "add" in names
    assert "mul" in names
    assert "sin" in names


def test_unknown_operator_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown operator"):
        fit({"x": [0.0, 1.0]}, [0.0, 1.0], operators=("mystery",))


def test_missing_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="target"):
        fit({"x": [0.0, 1.0]})


def test_feature_name_count_must_match_columns() -> None:
    with pytest.raises(ValueError, match="feature_names"):
        fit(np.zeros((4, 2)), np.zeros(4), feature_names=["x"])


def test_initial_expression_must_reference_a_feature() -> None:
    with pytest.raises(ValueError, match="not a feature"):
        fit({"x": [0.0, 1.0]}, [0.0, 1.0], initial_expression="z")


def test_decision_policy_requires_two_operators() -> None:
    with pytest.raises(ValueError, match="at least two operators"):
        fit(
            {"x": [0.0, 1.0]},
            [0.0, 1.0],
            operators=("mul",),
            policy="decision",
        )
