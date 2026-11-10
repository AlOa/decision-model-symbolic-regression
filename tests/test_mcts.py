import math

import numpy as np
import pytest

from symbolic_regression.data import Dataset
from symbolic_regression.decision.policy import PolicyResult, UniformOperatorPolicy
from symbolic_regression.expression import BinaryOp, Constant, UnaryOp, Variable, node_count, to_string
from symbolic_regression.metrics import evaluate_expression, reward
from symbolic_regression.result import StopReason
from symbolic_regression.search.mcts import MCTS, MCTSConfig, Node, _is_better


def quadratic_dataset() -> Dataset:
    x = np.linspace(-2.0, 2.0, 21)
    return Dataset.from_arrays({"x": x}, x**2)


def negative_dataset() -> Dataset:
    x = np.linspace(-2.0, -0.1, 11)
    return Dataset.from_arrays({"x": x}, x**2)


class FixedPolicy:
    name = "fixed"

    def __init__(self, probabilities: dict[str, float], source: str = "test") -> None:
        self.probabilities_map = probabilities
        self.source = source
        self.calls = 0

    def probabilities(self, context) -> PolicyResult:
        self.calls += 1
        return PolicyResult(
            probabilities=dict(self.probabilities_map),
            selected=next(iter(self.probabilities_map)),
            source=self.source,
        )


def config(**overrides) -> MCTSConfig:
    base = dict(
        iterations=1,
        max_nodes=200,
        max_depth=3,
        rollout_depth=0,
        top_applications=2,
        max_operands=64,
        fit_threshold=0.995,
        max_steps=12,
    )
    base.update(overrides)
    return MCTSConfig(**base)


def test_root_and_children_are_created() -> None:
    dataset = quadratic_dataset()
    mcts = MCTS(Variable("x"), dataset, FixedPolicy({"add": 1.0}), config())
    result = mcts.run()
    assert mcts.root is not None
    assert mcts.root.children
    assert result.history[0]["expanded_operator"] == "add"
    assert len(result.history[0]["selected_path"]) == 2
    assert all(child.depth == 1 for child in mcts.root.children)


def test_unvisited_children_are_selected_first() -> None:
    dataset = quadratic_dataset()
    mcts = MCTS(Variable("x"), dataset, FixedPolicy({"add": 1.0}), config())
    parent = Node(expression=Variable("x"), depth=0)
    visited_a = Node(expression=Variable("x"), parent=parent, depth=1, prior=0.9, visits=1, reward_sum=0.1)
    visited_b = Node(expression=Variable("x"), parent=parent, depth=1, prior=0.1, visits=1, reward_sum=0.9)
    unvisited = Node(expression=Variable("x"), parent=parent, depth=1, prior=0.5)
    parent.children = [visited_a, visited_b, unvisited]
    parent.visits = 2
    assert mcts._select_child(parent) is unvisited


def test_prior_influences_the_bound() -> None:
    dataset = quadratic_dataset()
    mcts = MCTS(Variable("x"), dataset, FixedPolicy({"add": 1.0}), config())
    parent = Node(expression=Variable("x"), depth=0, visits=10)
    high_prior = Node(expression=Variable("x"), parent=parent, depth=1, prior=0.9, visits=1, reward_sum=0.5)
    low_prior = Node(expression=Variable("x"), parent=parent, depth=1, prior=0.1, visits=1, reward_sum=0.5)
    parent.children = [low_prior, high_prior]
    assert mcts._select_child(parent) is high_prior


def test_exploration_constant_changes_selection() -> None:
    dataset = quadratic_dataset()
    parent = Node(expression=Variable("x"), depth=0, visits=11)
    exploit = Node(expression=Variable("x"), parent=parent, depth=1, prior=0.01, visits=10, reward_sum=9.0)
    explore = Node(expression=Variable("x"), parent=parent, depth=1, prior=0.99, visits=1, reward_sum=0.0)
    parent.children = [exploit, explore]
    greedy = MCTS(Variable("x"), dataset, FixedPolicy({"add": 1.0}), config(c_puct=0.0))
    curious = MCTS(Variable("x"), dataset, FixedPolicy({"add": 1.0}), config(c_puct=10.0))
    assert greedy._select_child(parent) is exploit
    assert curious._select_child(parent) is explore


def test_expansion_respects_depth_limit() -> None:
    dataset = quadratic_dataset()
    mcts = MCTS(
        Variable("x"),
        dataset,
        FixedPolicy({"add": 1.0}),
        config(iterations=4, max_depth=1),
    )
    mcts.run()
    assert mcts.root is not None
    assert all(child.depth <= 1 for child in mcts.root.children)


def test_invalid_operator_creates_no_children() -> None:
    dataset = negative_dataset()
    mcts = MCTS(
        Variable("x"),
        dataset,
        FixedPolicy({"log": 1.0}),
        config(iterations=1, max_depth=2),
    )
    result = mcts.run()
    assert mcts.root is not None
    assert mcts.root.children == []
    assert result.stop_reason is StopReason.NO_VALID_OPERATOR


def test_perfect_application_reaches_fit_threshold() -> None:
    dataset = quadratic_dataset()
    mcts = MCTS(
        Variable("x"),
        dataset,
        FixedPolicy({"mul": 1.0}),
        config(iterations=1, max_depth=2, rollout_depth=1),
    )
    result = mcts.run()
    assert result.best_expression is not None
    assert to_string(result.best_expression) == "(x * x)"
    assert result.best_reward == pytest.approx(1.0)
    assert result.stop_reason is StopReason.FIT_THRESHOLD


def test_better_fit_is_preferred_and_ties_break_by_complexity() -> None:
    dataset = quadratic_dataset()
    mcts = MCTS(Variable("x"), dataset, FixedPolicy({"add": 1.0}), config())
    simple = Node(expression=Variable("x"), evaluation=evaluate_expression(Variable("x"), dataset), depth=1)
    complex_node = Node(
        expression=BinaryOp("add", Variable("x"), Constant(100.0)),
        evaluation=evaluate_expression(BinaryOp("add", Variable("x"), Constant(100.0)), dataset),
        depth=1,
    )
    mcts._update_best(simple)
    best_before = mcts._best_node
    mcts._update_best(complex_node)
    assert mcts._best_node is best_before
    equal = Node(expression=Variable("x"), evaluation=evaluate_expression(Variable("x"), dataset), depth=1)
    assert not _is_better(equal, simple)


def test_backpropagation_updates_the_selected_path() -> None:
    dataset = quadratic_dataset()
    mcts = MCTS(
        Variable("x"),
        dataset,
        FixedPolicy({"add": 1.0}),
        config(iterations=3, max_depth=2, rollout_depth=0),
    )
    result = mcts.run()
    assert mcts.root is not None
    assert mcts.root.visits == 3
    assert result.decision_calls >= 1


def test_iterating_budget_stops_gracefully() -> None:
    dataset = quadratic_dataset()
    result = MCTS(
        Variable("x"), dataset, FixedPolicy({"add": 1.0}), config(iterations=1)
    ).run()
    assert result.iterations == 1
    assert result.stop_reason is StopReason.BUDGET_EXHAUSTED


def test_node_budget_stops_before_expansion() -> None:
    dataset = quadratic_dataset()
    result = MCTS(
        Variable("x"), dataset, FixedPolicy({"add": 1.0}), config(iterations=5, max_nodes=1)
    ).run()
    assert result.iterations == 0
    assert result.stop_reason is StopReason.BUDGET_EXHAUSTED


def test_time_budget_stops_immediately() -> None:
    dataset = quadratic_dataset()
    result = MCTS(
        Variable("x"),
        dataset,
        FixedPolicy({"add": 1.0}),
        config(iterations=5, time_limit_seconds=1e-9),
    ).run()
    assert result.iterations == 0


def test_fallback_policy_is_counted() -> None:
    dataset = quadratic_dataset()
    result = MCTS(
        Variable("x"),
        dataset,
        FixedPolicy({"add": 1.0}, source="fallback"),
        config(iterations=1),
    ).run()
    assert result.fallback_calls >= 1
    assert result.history[0]["policy_source"] == "fallback"


def test_no_feasible_expression_is_reported() -> None:
    dataset = negative_dataset()
    initial = UnaryOp("log", Variable("x"))
    result = MCTS(
        initial,
        dataset,
        FixedPolicy({"sin": 1.0}),
        config(iterations=1, max_depth=2),
    ).run()
    assert result.best_expression is None
    assert result.stop_reason is StopReason.NO_VALID_OPERATOR


def test_search_trace_fields() -> None:
    dataset = quadratic_dataset()
    result = MCTS(Variable("x"), dataset, FixedPolicy({"mul": 1.0}), config()).run()
    record = result.history[0]
    for key in (
        "iteration",
        "selected_path",
        "expanded_operator",
        "expanded_operand",
        "rollout_reward",
        "best_expression",
        "best_reward",
        "policy_source",
    ):
        assert key in record


def test_policy_cache_avoids_duplicate_calls() -> None:
    dataset = quadratic_dataset()
    policy = FixedPolicy({"add": 1.0})
    mcts = MCTS(Variable("x"), dataset, policy, config())
    node = Node(expression=Variable("x"), depth=0)
    mcts._policy_result(node)
    mcts._policy_result(node)
    assert policy.calls == 1
    assert mcts._decision_calls == 1


def test_uniform_policy_integration() -> None:
    dataset = quadratic_dataset()
    result = MCTS(
        Variable("x"), dataset, UniformOperatorPolicy(), config(iterations=1)
    ).run()
    assert result.decision_calls >= 1
    assert result.fallback_calls == 0
    assert result.history[0]["policy_source"] == "uniform"


def test_reward_is_monotonic_in_fit() -> None:
    dataset = quadratic_dataset()
    exact = evaluate_expression(BinaryOp("mul", Variable("x"), Variable("x")), dataset)
    worse = evaluate_expression(Variable("x"), dataset)
    assert reward(exact) > reward(worse)
    assert node_count(BinaryOp("mul", Variable("x"), Variable("x"))) == 3


@pytest.mark.parametrize("name", ["linear", "quadratic", "sine", "composite"])
def test_rollout_reward_uses_each_suite_tier(name: str) -> None:
    from symbolic_regression.experiments.datasets import generate_dataset

    dataset = generate_dataset(name, n_samples=16, seed=2, noise_std=0.05)
    result = MCTS(
        Variable("x"),
        dataset,
        UniformOperatorPolicy(),
        config(iterations=1, max_depth=2, rollout_depth=1),
    ).run()
    assert result.best_evaluation is not None
    assert 0.0 <= result.best_reward <= 1.0


def test_greedy_mode_follows_a_single_path() -> None:
    dataset = quadratic_dataset()
    mcts = MCTS(
        Variable("x"),
        dataset,
        FixedPolicy({"mul": 1.0}),
        config(iterations=10, greedy=True, max_depth=3, max_steps=3),
    )
    result = mcts.run()
    assert result.best_expression is not None
    assert to_string(result.best_expression) == "(x * x)"
    assert result.stop_reason is StopReason.FIT_THRESHOLD
    assert mcts.root is not None
    assert len(mcts.root.children) == 1


