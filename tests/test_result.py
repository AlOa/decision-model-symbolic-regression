from symbolic_regression.expression import Constant
from symbolic_regression.metrics import Evaluation
from symbolic_regression.result import (
    SearchResult,
    StopReason,
    check_stopping,
    reward_of,
)


def evaluation(r2: float, nmse: float = 0.0) -> Evaluation:
    return Evaluation(None, None, nmse, nmse, r2, True)


def test_fit_threshold_stops() -> None:
    decision = check_stopping(1, 10, evaluation(0.99), 0.95)
    assert decision.should_stop
    assert decision.reason is StopReason.FIT_THRESHOLD


def test_max_steps_stops() -> None:
    decision = check_stopping(10, 10, evaluation(0.1), 0.95)
    assert decision.should_stop
    assert decision.reason is StopReason.MAX_STEPS


def test_no_valid_operator_stops() -> None:
    decision = check_stopping(1, 10, evaluation(0.1), 0.95, has_valid_actions=False)
    assert decision.should_stop
    assert decision.reason is StopReason.NO_VALID_OPERATOR


def test_continue_when_no_condition_holds() -> None:
    decision = check_stopping(1, 10, evaluation(0.1), 0.95)
    assert not decision.should_stop


def test_search_result_serialization() -> None:
    result = SearchResult(
        best_expression=Constant(3.5),
        best_evaluation=evaluation(1.0),
        best_reward=1.0,
        steps=2,
        stop_reason=StopReason.FIT_THRESHOLD,
    )
    payload = result.to_dict()
    assert payload["best_expression"] == "3.5"
    assert payload["simplified_expression"] == "7/2"
    assert payload["stop_reason"] == "fit_threshold"
    assert reward_of(evaluation(1.0)) == 1.0


def test_search_result_simplification_is_best_effort() -> None:
    result = SearchResult(
        best_expression=None,
        best_evaluation=None,
        best_reward=0.0,
        steps=0,
        stop_reason=StopReason.ERROR,
    )
    assert result.simplified_expression is None
    assert result.to_dict()["simplified_expression"] is None
