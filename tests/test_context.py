import json

import numpy as np

from symbolic_regression.context import build_context
from symbolic_regression.data import Dataset
from symbolic_regression.expression import BinaryOp, Constant, Variable


def make_dataset() -> Dataset:
    x = np.linspace(-2, 2, 20)
    return Dataset.from_arrays({"x": x}, x**2)


def test_context_contains_required_fields() -> None:
    dataset = make_dataset()
    context = build_context(BinaryOp("mul", Variable("x"), Variable("x")), dataset, step=0)
    payload = context.to_dict()
    assert "schema_version" in payload
    assert payload["step"] == 0
    assert payload["expression"] == "(x * x)"
    assert set(payload["features"]) == {"x"}
    assert len(payload["target"]) == 20
    assert payload["residual"]["available"] is True
    assert payload["history"] == []


def test_context_serialization_is_deterministic() -> None:
    dataset = make_dataset()
    expr = BinaryOp("mul", Variable("x"), Variable("x"))
    first = build_context(expr, dataset, step=2, seed=7)
    second = build_context(expr, dataset, step=2, seed=7)
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()


def test_non_finite_residual_is_marked_unavailable() -> None:
    dataset = make_dataset()
    context = build_context(BinaryOp("div", Constant(1.0), Constant(0.0)), dataset, step=1)
    assert context.to_dict()["residual"]["available"] is False
    assert context.to_dict()["residual"]["values"] == []


def test_history_is_included() -> None:
    dataset = make_dataset()
    history = [{"operator": "mul", "confidence": 0.9}]
    context = build_context(Variable("x"), dataset, step=1, history=history)
    assert context.to_dict()["history"] == history


def test_context_builds_for_suite_dataset(suite_dataset) -> None:
    name, label, dataset = suite_dataset
    assert name and label
    context = build_context(Variable(dataset.feature_names[0]), dataset, step=0)
    payload = context.to_dict()
    assert payload["target"]
    assert payload["expression"] == dataset.feature_names[0]
    assert payload["residual"]["available"] is True
