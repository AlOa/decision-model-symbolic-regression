"""Monte Carlo Tree Search over incremental expression construction."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

from ..context import DecisionContext, build_context
from ..data import Dataset
from ..decision.policy import OperatorPolicy, PolicyResult
from ..expression import (
    BINARY_OPERATORS,
    OPERATORS,
    UNARY_OPERATORS,
    Expr,
    UnaryOp,
    BinaryOp,
    Constant,
    to_string,
    node_count,
)
from ..metrics import Evaluation, evaluate_expression, reward
from ..result import SearchResult, StopReason
from .operands import build_operand_bank


@dataclass(slots=True)
class MCTSConfig:
    iterations: int = 50
    max_nodes: int = 500
    max_depth: int = 6
    rollout_depth: int = 3
    c_puct: float = 1.4
    top_applications: int = 2
    max_operands: int = 256
    fit_threshold: float = 0.995
    max_steps: int = 12
    time_limit_seconds: float = 300.0
    seed: int = 0
    greedy: bool = False
    operators: tuple[str, ...] | None = None
    context_rows: int = 24
    residual_values: int = 16

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if self.max_depth < 1:
            raise ValueError("max_depth must be positive")
        if self.rollout_depth < 0:
            raise ValueError("rollout_depth must be non-negative")
        if self.top_applications < 1:
            raise ValueError("top_applications must be positive")
        if not (0.0 <= self.fit_threshold <= 1.0):
            raise ValueError("fit_threshold must be between 0 and 1")
        if self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if self.context_rows < 1:
            raise ValueError("context_rows must be positive")
        if self.residual_values < 1:
            raise ValueError("residual_values must be positive")
        if self.operators is not None:
            names = tuple(dict.fromkeys(str(name) for name in self.operators))
            if not names:
                raise ValueError("operators must contain at least one operator")
            unknown = [name for name in names if name not in OPERATORS]
            if unknown:
                raise ValueError(f"Unknown operators: {', '.join(unknown)}")
            self.operators = names


@dataclass(frozen=True, slots=True)
class Action:
    operator: str
    expression: Expr
    operand: str | None = None
    orientation: str = "unary"


@dataclass(slots=True)
class Node:
    expression: Expr
    parent: "Node | None" = None
    action: Action | None = None
    depth: int = 0
    children: list["Node"] = field(default_factory=list)
    prior: float = 0.0
    visits: int = 0
    reward_sum: float = 0.0
    evaluation: Evaluation | None = None
    terminal: bool = False
    expanded: bool = False
    policy_result: PolicyResult | None = None

    @property
    def mean_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.reward_sum / self.visits


class MCTS:
    """PUCT search whose priors come from a decision-model operator policy."""

    def __init__(
        self,
        initial_expression: Expr,
        dataset: Dataset,
        policy: OperatorPolicy,
        config: MCTSConfig | None = None,
    ) -> None:
        self.initial_expression = initial_expression
        self.dataset = dataset
        self.policy = policy
        self.config = config or MCTSConfig()
        self._evaluation_cache: dict[str, Evaluation] = {}
        self._policy_cache: dict[str, PolicyResult] = {}
        self._decision_records: list[dict[str, object]] = []
        self._iteration_records: list[dict[str, object]] = []
        self._decision_calls = 0
        self._fallback_calls = 0
        self._node_count = 0
        self.operators = (
            tuple(self.config.operators)
            if self.config.operators is not None
            else tuple(OPERATORS)
        )
        self._operator_set = frozenset(self.operators)
        binary_names = tuple(
            name for name in self.operators if name in BINARY_OPERATORS
        )
        unary_names = tuple(
            name for name in self.operators if name in UNARY_OPERATORS
        )
        self._operand_bank = build_operand_bank(
            dataset.feature_names,
            unary_names=unary_names,
            binary_names=binary_names,
            max_operands=self.config.max_operands,
        )
        self._best_node: Node | None = None
        self.root: Node | None = None

    # ------------------------------------------------------------------ public

    def run(self) -> SearchResult:
        started = time.perf_counter()
        root = Node(
            expression=self.initial_expression,
            depth=0,
            evaluation=self._evaluate(self.initial_expression),
        )
        self._node_count = 1
        self.root = root
        self._best_node = root if _has_valid_evaluation(root) else None
        if self.config.greedy:
            return self._run_greedy(root, started)
        stop_reason = StopReason.BUDGET_EXHAUSTED
        iterations_done = 0

        for iteration in range(self.config.iterations):
            if time.perf_counter() - started >= self.config.time_limit_seconds:
                stop_reason = StopReason.BUDGET_EXHAUSTED
                break
            if self._node_count >= self.config.max_nodes:
                stop_reason = StopReason.BUDGET_EXHAUSTED
                break

            selected = self._select(root)
            path = _path_to(selected)
            rollout_start = selected
            if (
                not selected.expanded
                and not selected.terminal
                and selected.depth < self.config.max_depth
            ):
                policy_result = self._policy_result(selected)
                children = self._generate_children(selected, policy_result)
                self._node_count += len(children)
                if not children:
                    selected.terminal = True
                else:
                    rollout_start = max(children, key=lambda child: child.prior)
                    path = [*path, rollout_start]

            rollout_best = self._rollout(rollout_start)
            value = (
                reward(rollout_best.evaluation)
                if rollout_best is not None and _has_valid_evaluation(rollout_best)
                else 0.0
            )
            self._update_best(rollout_start)
            self._update_best(rollout_best)

            cursor: Node | None = rollout_start
            while cursor is not None:
                cursor.visits += 1
                cursor.reward_sum += value
                cursor = cursor.parent

            iterations_done = iteration + 1
            self._iteration_records.append(
                {
                    "iteration": iteration,
                    "selected_path": [to_string(node.expression) for node in path],
                    "expanded_operator": (
                        rollout_start.action.operator if rollout_start.action else None
                    ),
                    "expanded_operand": (
                        rollout_start.action.operand if rollout_start.action else None
                    ),
                    "expanded_orientation": (
                        rollout_start.action.orientation if rollout_start.action else None
                    ),
                    "rollout_reward": float(value),
                    "rollout_steps": int(rollout_best.depth if rollout_best else 0),
                    "best_expression": (
                        to_string(self._best_node.expression) if self._best_node else None
                    ),
                    "best_reward": self._best_reward(),
                    "policy_source": (
                        rollout_start.policy_result.source
                        if rollout_start.policy_result
                        else None
                    ),
                    "decision_calls": self._decision_calls,
                    "fallback_calls": self._fallback_calls,
                }
            )

            if self._best_node is not None and _has_valid_evaluation(self._best_node):
                if self._best_node.evaluation.r2 >= self.config.fit_threshold:
                    stop_reason = StopReason.FIT_THRESHOLD
                    break
            if self._best_node is not None and self._best_node.depth >= self.config.max_steps:
                stop_reason = StopReason.MAX_STEPS
                break
            if root.terminal and not root.children:
                stop_reason = StopReason.NO_VALID_OPERATOR
                break
            if root.expanded and not root.children:
                stop_reason = StopReason.NO_VALID_OPERATOR
                break

        elapsed = time.perf_counter() - started
        if root.terminal and not root.children:
            stop_reason = StopReason.NO_VALID_OPERATOR
        elif stop_reason is StopReason.BUDGET_EXHAUSTED and self._best_node is not None:
            if _has_valid_evaluation(self._best_node) and (
                self._best_node.evaluation.r2 >= self.config.fit_threshold
            ):
                stop_reason = StopReason.FIT_THRESHOLD
            elif self._best_node.depth >= self.config.max_steps:
                stop_reason = StopReason.MAX_STEPS

        return SearchResult(
            best_expression=self._best_node.expression if self._best_node else None,
            best_evaluation=self._best_node.evaluation if self._best_node else None,
            best_reward=self._best_reward(),
            steps=self._best_node.depth if self._best_node else 0,
            stop_reason=stop_reason,
            history=list(self._iteration_records),
            decision_records=list(self._decision_records),
            decision_calls=self._decision_calls,
            fallback_calls=self._fallback_calls,
            iterations=iterations_done,
            elapsed_seconds=elapsed,
        )

    def _run_greedy(self, root: Node, started: float) -> SearchResult:
        current = root
        stop_reason = StopReason.BUDGET_EXHAUSTED
        iterations_done = 0
        for step in range(self.config.max_steps):
            if time.perf_counter() - started >= self.config.time_limit_seconds:
                break
            if self._node_count >= self.config.max_nodes:
                break
            if current.depth >= self.config.max_depth:
                break
            policy_result = self._policy_result(current)
            order = sorted(
                policy_result.probabilities,
                key=lambda name: (
                    -policy_result.probabilities[name],
                    list(policy_result.probabilities).index(name),
                ),
            )
            chosen: tuple[Action, Evaluation] | None = None
            for operator in order:
                found = self._applications_for_operator(current, operator, limit=1)
                if found:
                    chosen = found[0]
                    break
            if chosen is None:
                current.terminal = True
                stop_reason = StopReason.NO_VALID_OPERATOR
                break
            action, evaluation = chosen
            child = Node(
                expression=action.expression,
                parent=current,
                action=action,
                depth=current.depth + 1,
                prior=1.0,
                evaluation=evaluation,
                policy_result=policy_result,
            )
            current.children = [child]
            current.expanded = True
            current.policy_result = policy_result
            self._node_count += 1
            self._update_best(child)
            current = child
            iterations_done = step + 1
            self._iteration_records.append(
                {
                    "iteration": step,
                    "selected_path": [to_string(node.expression) for node in _path_to(current)],
                    "expanded_operator": action.operator,
                    "expanded_operand": action.operand,
                    "expanded_orientation": action.orientation,
                    "rollout_reward": reward(evaluation),
                    "rollout_steps": current.depth,
                    "best_expression": (
                        to_string(self._best_node.expression) if self._best_node else None
                    ),
                    "best_reward": self._best_reward(),
                    "policy_source": policy_result.source,
                    "decision_calls": self._decision_calls,
                    "fallback_calls": self._fallback_calls,
                }
            )
            if self._best_node is not None and _has_valid_evaluation(self._best_node):
                if self._best_node.evaluation.r2 >= self.config.fit_threshold:
                    stop_reason = StopReason.FIT_THRESHOLD
                    break
            if current.depth >= self.config.max_steps:
                stop_reason = StopReason.MAX_STEPS
                break
        elapsed = time.perf_counter() - started
        if stop_reason is StopReason.BUDGET_EXHAUSTED and self._best_node is not None:
            if _has_valid_evaluation(self._best_node) and (
                self._best_node.evaluation.r2 >= self.config.fit_threshold
            ):
                stop_reason = StopReason.FIT_THRESHOLD
            elif self._best_node.depth >= self.config.max_steps:
                stop_reason = StopReason.MAX_STEPS
        return SearchResult(
            best_expression=self._best_node.expression if self._best_node else None,
            best_evaluation=self._best_node.evaluation if self._best_node else None,
            best_reward=self._best_reward(),
            steps=self._best_node.depth if self._best_node else 0,
            stop_reason=stop_reason,
            history=list(self._iteration_records),
            decision_records=list(self._decision_records),
            decision_calls=self._decision_calls,
            fallback_calls=self._fallback_calls,
            iterations=iterations_done,
            elapsed_seconds=elapsed,
        )

    # --------------------------------------------------------------- selection

    def _select(self, root: Node) -> Node:
        node = root
        while node.expanded and node.children and node.depth < self.config.max_depth:
            node = self._select_child(node)
        return node

    def _select_child(self, node: Node) -> Node:
        unvisited = [child for child in node.children if child.visits == 0]
        if unvisited:
            return max(unvisited, key=lambda child: (child.prior, -node.children.index(child)))
        parent_visits = max(1, node.visits)
        return max(
            node.children,
            key=lambda child: child.mean_reward
            + self.config.c_puct
            * child.prior
            * math.sqrt(parent_visits)
            / (1 + child.visits),
        )

    # ---------------------------------------------------------------- expansion

    def _policy_result(self, node: Node) -> PolicyResult:
        context = self._context(node)
        key = context.to_json()
        if key in self._policy_cache:
            return self._policy_cache[key]
        result = self.policy.probabilities(context)
        self._decision_calls += 1
        if result.source == "fallback":
            self._fallback_calls += 1
        self._policy_cache[key] = result
        record = {
            "step": node.depth,
            "expression": to_string(node.expression),
            "candidates": list(result.probabilities),
            **result.to_trace(),
        }
        self._decision_records.append(record)
        return result

    def _generate_children(
        self, node: Node, policy_result: PolicyResult
    ) -> list[Node]:
        applications: dict[str, list[tuple[Action, Evaluation]]] = {}
        for operator, probability in policy_result.probabilities.items():
            if probability <= 0:
                continue
            found = self._applications_for_operator(node, operator, limit=self.config.top_applications)
            if found:
                applications[operator] = found
        if not applications:
            node.children = []
            node.expanded = True
            node.policy_result = policy_result
            return []
        total_probability = sum(
            max(0.0, policy_result.probabilities[operator]) for operator in applications
        )
        if total_probability <= 0:
            weight = 1.0 / len(applications)
            weights = {operator: weight for operator in applications}
        else:
            weights = {
                operator: max(0.0, policy_result.probabilities[operator]) / total_probability
                for operator in applications
            }
        children: list[Node] = []
        for operator, found in applications.items():
            share = weights[operator] / len(found)
            for action, evaluation in found:
                children.append(
                    Node(
                        expression=action.expression,
                        parent=node,
                        action=action,
                        depth=node.depth + 1,
                        prior=share,
                        evaluation=evaluation,
                        policy_result=policy_result,
                    )
                )
        total_prior = sum(child.prior for child in children)
        if total_prior > 0:
            for child in children:
                child.prior /= total_prior
        else:
            for child in children:
                child.prior = 1.0 / len(children)
        node.children = children
        node.expanded = True
        node.policy_result = policy_result
        return children

    def _applications_for_operator(
        self, node: Node, operator: str, limit: int
    ) -> list[tuple[Action, Evaluation]]:
        if operator not in self._operator_set:
            return []
        found: list[tuple[Action, Evaluation]] = []
        if operator in BINARY_OPERATORS:
            for operand in self._operand_bank:
                if self._is_neutral(operator, operand):
                    continue
                for orientation in ("right", "left"):
                    if orientation == "right":
                        expression = BinaryOp(operator, node.expression, operand)
                    else:
                        expression = BinaryOp(operator, operand, node.expression)
                    evaluation = self._evaluate(expression)
                    if evaluation.valid:
                        found.append(
                            (
                                Action(
                                    operator=operator,
                                    expression=expression,
                                    operand=to_string(operand),
                                    orientation=orientation,
                                ),
                                evaluation,
                            )
                        )
        else:
            expression = UnaryOp(operator, node.expression)
            evaluation = self._evaluate(expression)
            if evaluation.valid:
                found.append(
                    (
                        Action(
                            operator=operator,
                            expression=expression,
                            operand=None,
                            orientation="unary",
                        ),
                        evaluation,
                    )
                )
        deduplicated: dict[str, tuple[Action, Evaluation]] = {}
        for action, evaluation in found:
            deduplicated.setdefault(to_string(action.expression), (action, evaluation))
        ordered = sorted(
            deduplicated.values(),
            key=lambda item: (
                -reward(item[1]),
                node_count(item[0].expression),
                to_string(item[0].expression),
            ),
        )
        return ordered[:limit]

    def _is_neutral(self, operator: str, operand: Expr) -> bool:
        if not isinstance(operand, Constant):
            return False
        value = float(operand.value)
        if operator in ("add", "sub") and value == 0.0:
            return True
        if operator in ("mul", "div") and value == 1.0:
            return True
        if operator == "mul" and value == 0.0:
            return True
        return False

    # ------------------------------------------------------------------ rollout

    def _rollout(self, start: Node) -> Node:
        best = start if _has_valid_evaluation(start) else None
        current = start
        for _ in range(self.config.rollout_depth):
            if current.depth >= self.config.max_depth:
                break
            policy_result = self._policy_result(current)
            chosen: tuple[Action, Evaluation] | None = None
            order = sorted(
                policy_result.probabilities,
                key=lambda name: (
                    -policy_result.probabilities[name],
                    list(policy_result.probabilities).index(name),
                ),
            )
            for operator in order:
                found = self._applications_for_operator(current, operator, limit=1)
                if found:
                    chosen = found[0]
                    break
            if chosen is None:
                break
            action, evaluation = chosen
            child = Node(
                expression=action.expression,
                parent=current,
                action=action,
                depth=current.depth + 1,
                evaluation=evaluation,
                policy_result=policy_result,
            )
            if _has_valid_evaluation(child) and (
                best is None or _is_better(child, best)
            ):
                best = child
            if evaluation.valid and evaluation.r2 is not None:
                if evaluation.r2 >= self.config.fit_threshold:
                    break
            current = child
        return best if best is not None else start

    # ------------------------------------------------------------------ helpers

    def _context(self, node: Node) -> DecisionContext:
        return build_context(
            node.expression,
            self.dataset,
            step=node.depth,
            history=_history(node),
            context_rows=self.config.context_rows,
            residual_values=self.config.residual_values,
            seed=self.config.seed,
        )

    def _evaluate(self, expression: Expr) -> Evaluation:
        key = to_string(expression)
        cached = self._evaluation_cache.get(key)
        if cached is None:
            cached = evaluate_expression(expression, self.dataset)
            self._evaluation_cache[key] = cached
        return cached

    def _update_best(self, node: Node | None) -> None:
        if node is None or not _has_valid_evaluation(node):
            return
        if self._best_node is None or _is_better(node, self._best_node):
            self._best_node = node

    def _best_reward(self) -> float:
        if self._best_node is None or not _has_valid_evaluation(self._best_node):
            return 0.0
        return reward(self._best_node.evaluation)


def _has_valid_evaluation(node: Node) -> bool:
    return node.evaluation is not None and node.evaluation.valid


def _is_better(candidate: Node, current: Node) -> bool:
    if not _has_valid_evaluation(candidate):
        return False
    if not _has_valid_evaluation(current):
        return True
    candidate_reward = reward(candidate.evaluation)
    current_reward = reward(current.evaluation)
    if candidate_reward > current_reward + 1e-12:
        return True
    if abs(candidate_reward - current_reward) <= 1e-12:
        candidate_complexity = node_count(candidate.expression)
        current_complexity = node_count(current.expression)
        if candidate_complexity < current_complexity:
            return True
        if candidate_complexity == current_complexity:
            return to_string(candidate.expression) < to_string(current.expression)
    return False


def _path_to(node: Node) -> list[Node]:
    path: list[Node] = []
    cursor: Node | None = node
    while cursor is not None:
        path.append(cursor)
        cursor = cursor.parent
    return list(reversed(path))


def _history(node: Node, limit: int = 8) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    cursor: Node | None = node
    while cursor is not None and cursor.parent is not None:
        if cursor.action is not None:
            record: dict[str, object] = {
                "operator": cursor.action.operator,
                "operand": cursor.action.operand,
                "orientation": cursor.action.orientation,
                "prior": round(cursor.prior, 6),
            }
            if cursor.policy_result is not None:
                record["confidence"] = cursor.policy_result.confidence
                record["source"] = cursor.policy_result.source
            records.append(record)
        cursor = cursor.parent
    records.reverse()
    return records[-limit:]
