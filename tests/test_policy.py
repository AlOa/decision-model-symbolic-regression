import numpy as np
import pytest

from symbolic_regression.context import build_context
from symbolic_regression.data import Dataset
from symbolic_regression.decision.client import OllamaConnectionError, SystemOneChoice
from symbolic_regression.decision.policy import (
    DEFAULT_OPERATORS,
    OfflineOperatorPolicy,
    OllamaOperatorPolicy,
    PolicyResult,
    UniformOperatorPolicy,
    make_policy,
    select_highest,
)
from symbolic_regression.expression import Variable


def make_context():
    x = np.linspace(-1.0, 1.0, 8)
    dataset = Dataset.from_arrays({"x": x}, x**2)
    return build_context(Variable("x"), dataset, step=0)


class FakeClient:
    def __init__(self, choice=None, error=None):
        self.choice_result = choice
        self.error = error
        self.calls = []

    def choice(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.choice_result


def choice(probabilities, selected="add", confidence=0.8):
    return SystemOneChoice(
        selected=selected,
        probabilities=probabilities,
        confidence=confidence,
        raw={"choice": selected},
    )


def test_highest_probability_wins_even_if_model_choice_differs() -> None:
    client = FakeClient(choice({"add": 0.1, "mul": 0.6, "sin": 0.3}, selected="add"))
    policy = OllamaOperatorPolicy(client, "d3:4b")
    result = policy.probabilities(make_context())
    assert result.selected == "mul"
    assert result.source == "ollama"
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_ties_are_broken_by_catalogue_order() -> None:
    client = FakeClient(choice({"add": 0.5, "mul": 0.5}, selected="mul"))
    policy = OllamaOperatorPolicy(client, "d3:4b", operators=("add", "mul"))
    result = policy.probabilities(make_context())
    assert result.selected == "add"


def test_candidate_order_is_preserved() -> None:
    client = FakeClient(choice({"sin": 0.7, "cos": 0.3}, selected="sin"))
    policy = OllamaOperatorPolicy(client, "d3:4b", operators=("sin", "cos"))
    policy.probabilities(make_context())
    assert list(client.calls[0]["criteria"]) == ["sin", "cos"]


def test_fallback_on_connection_error_is_reported_and_uniform() -> None:
    client = FakeClient(error=OllamaConnectionError("down"))
    policy = OllamaOperatorPolicy(client, "d3:4b")
    result = policy.probabilities(make_context())
    assert result.source == "fallback"
    assert result.error == "down"
    assert result.selected == DEFAULT_OPERATORS[0]
    assert len(set(result.probabilities.values())) == 1


def test_cache_avoids_second_model_call() -> None:
    client = FakeClient(choice({"add": 0.6, "mul": 0.4}, selected="add"))
    policy = OllamaOperatorPolicy(client, "d3:4b", operators=("add", "mul"))
    context = make_context()
    first = policy.probabilities(context)
    second = policy.probabilities(context)
    assert len(client.calls) == 1
    assert first.source == "ollama"
    assert second.source == "cache"


def test_uniform_policy_is_equal() -> None:
    result = UniformOperatorPolicy().probabilities(make_context())
    values = list(result.probabilities.values())
    assert all(value == pytest.approx(values[0]) for value in values)


def test_offline_policy_is_deterministic() -> None:
    context = make_context()
    first = OfflineOperatorPolicy(seed=7).probabilities(context)
    second = OfflineOperatorPolicy(seed=7).probabilities(context)
    assert first.probabilities == second.probabilities
    assert first.selected == second.selected


def test_trace_serialization_contains_required_fields() -> None:
    result = PolicyResult(
        probabilities={"add": 1.0},
        selected="add",
        source="ollama",
        confidence=0.5,
        model="d3:4b",
    )
    trace = result.to_trace()
    assert trace["model"] == "d3:4b"
    assert trace["selected"] == "add"
    assert trace["probabilities"] == {"add": 1.0}
    assert trace["confidence"] == 0.5
    assert "elapsed_seconds" in trace


def test_make_policy_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown policy"):
        make_policy("mystery")


def test_select_highest_empty_is_rejected() -> None:
    with pytest.raises(ValueError):
        select_highest({}, ())


def test_decision_context_from_noisy_dataset_produces_valid_question() -> None:
    from symbolic_regression.experiments.datasets import generate_dataset

    dataset = generate_dataset("sine", n_samples=16, seed=1, noise_std=0.1)
    context = build_context(Variable("x"), dataset, step=0)
    client = FakeClient(choice({"add": 0.5, "sin": 0.5}, selected="add"))
    policy = OllamaOperatorPolicy(client, "d3:4b", operators=("add", "sin"))
    result = policy.probabilities(context)
    assert result.source == "ollama"
    state = client.calls[0]["state"]
    assert state["features"]["x"]
    assert state["residual"]["available"] is True


def test_fallback_trace_contains_error() -> None:
    client = FakeClient(error=OllamaConnectionError("down"))
    policy = OllamaOperatorPolicy(client, "d3:4b")
    trace = policy.probabilities(make_context()).to_trace()
    assert trace["source"] == "fallback"
    assert trace["error"] == "down"
    assert trace["selected"] is not None

