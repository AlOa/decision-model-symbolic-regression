import json
from pathlib import Path

import pytest

from symbolic_regression.config import RunConfig
from symbolic_regression.experiments.runner import (
    RunnerError,
    compare_experiments,
    run_experiment,
)
from symbolic_regression.metrics import evaluate_expression


def offline_config(tmp_path: Path, **overrides) -> RunConfig:
    base = dict(
        function="quadratic",
        policy="offline",
        iterations=2,
        max_nodes=100,
        max_depth=2,
        rollout_depth=1,
        samples=16,
        noise_std=0.0,
        output=str(tmp_path),
        run_name="run",
        seed=1,
    )
    base.update(overrides)
    return RunConfig(**base)


def test_offline_run_writes_artifacts(tmp_path) -> None:
    outcome = run_experiment(offline_config(tmp_path))
    assert outcome.artifacts.result.exists()
    assert outcome.artifacts.trace.exists()
    assert outcome.artifacts.summary.exists()
    payload = json.loads(outcome.artifacts.result.read_text(encoding="utf-8"))
    assert payload["run"]["policy"] == "offline"
    assert payload["result"]["best_expression"]
    assert payload["result"]["ground_truth"]
    assert payload["dataset"]["metadata"]["noise_mode"] == "std"
    fresh = evaluate_expression(outcome.result.best_expression, outcome.dataset)
    assert fresh.to_dict() == outcome.result.best_evaluation.to_dict()
    summary = outcome.artifacts.summary.read_text(encoding="utf-8")
    for field in ("Recovered expression", "MSE", "NMSE", "R-squared", "Stop reason", "Noise"):
        assert field in summary


def test_comparison_writes_report(tmp_path) -> None:
    config = offline_config(
        tmp_path,
        comparison_policies=("offline", "uniform"),
        run_name="cmp",
    )
    comparison = compare_experiments(config)
    assert comparison.comparison_path.exists()
    payload = json.loads(comparison.comparison_path.read_text(encoding="utf-8"))
    assert set(payload["results"]) == {"offline", "uniform"}
    assert (comparison.directory / "comparison.md").exists()
    assert len(comparison.outcomes) == 2
    for outcome in comparison.outcomes:
        assert outcome.artifacts.result.exists()


def _strip_dynamic(value):
    if isinstance(value, dict):
        return {
            key: _strip_dynamic(item)
            for key, item in value.items()
            if key not in {"created_at", "elapsed_seconds", "output_dir"}
        }
    if isinstance(value, list):
        return [_strip_dynamic(item) for item in value]
    return value


def test_offline_run_is_reproducible(tmp_path) -> None:
    config = offline_config(tmp_path, iterations=2, max_depth=2)
    first = run_experiment(config)
    first_result = _strip_dynamic(json.loads(first.artifacts.result.read_text(encoding="utf-8")))
    first_trace = _strip_dynamic(json.loads(first.artifacts.trace.read_text(encoding="utf-8")))
    second = run_experiment(config)
    second_result = _strip_dynamic(json.loads(second.artifacts.result.read_text(encoding="utf-8")))
    second_trace = _strip_dynamic(json.loads(second.artifacts.trace.read_text(encoding="utf-8")))
    assert first_result == second_result
    assert first_trace == second_trace


def test_csv_input_path_is_supported(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n0,0\n1,1\n2,4\n3,9\n", encoding="utf-8")
    config = offline_config(
        tmp_path,
        function=None,
        csv=str(csv_path),
        target="y",
        run_name="csv-run",
    )
    outcome = run_experiment(config)
    assert outcome.dataset.metadata["source"] == "csv"
    assert outcome.artifacts.result.exists()


def test_decision_policy_without_server_is_actionable(tmp_path) -> None:
    config = offline_config(
        tmp_path,
        policy="decision",
        base_url="http://127.0.0.1:9",
        request_timeout=0.2,
        request_retries=0,
    )
    with pytest.raises(RunnerError, match="offline"):
        run_experiment(config)


def test_runner_rejects_missing_dataset_configuration() -> None:
    with pytest.raises(ValueError, match="Provide a suite function"):
        RunConfig()
