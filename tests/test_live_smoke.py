import os

import pytest

from symbolic_regression.config import RunConfig
from symbolic_regression.context import build_context
from symbolic_regression.decision.client import OllamaClient
from symbolic_regression.experiments.datasets import generate_dataset
from symbolic_regression.experiments.runner import run_experiment
from symbolic_regression.expression import Variable

pytestmark = pytest.mark.skipif(
    os.environ.get("OLLAMA_LIVE") != "1",
    reason="set OLLAMA_LIVE=1 to run live Ollama smoke tests",
)

MODEL = os.environ.get("OLLAMA_MODEL", "d3:4b")


def test_live_systemone_choice() -> None:
    with OllamaClient() as client:
        info = client.readiness(MODEL)
        assert "decision" in info.capabilities
        dataset = generate_dataset("sine", n_samples=16, seed=1, noise_std=0.05)
        context = build_context(Variable("x"), dataset, step=0)
        choice = client.choice(
            MODEL,
            context.to_dict(),
            "Choose the operator to add next.",
            {"sin": "sine", "cos": "cosine", "add": "addition"},
        )
    assert choice.selected in {"sin", "cos", "add"}
    assert set(choice.probabilities) == {"sin", "cos", "add"}
    assert sum(choice.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_live_short_experiment(tmp_path) -> None:
    config = RunConfig(
        function="sine",
        policy="decision",
        model=MODEL,
        iterations=1,
        max_nodes=20,
        max_depth=1,
        rollout_depth=0,
        samples=12,
        noise_std=0.05,
        output=str(tmp_path),
        run_name="live",
        seed=1,
        time_limit_seconds=60.0,
    )
    outcome = run_experiment(config)
    assert outcome.result.best_evaluation is not None
    assert outcome.result.decision_calls >= 1
    assert outcome.artifacts.result.exists()


def test_live_cli_short_run(tmp_path) -> None:
    from symbolic_regression.cli import main

    code = main(
        [
            "run",
            "--function",
            "sine",
            "--model",
            MODEL,
            "--iterations",
            "1",
            "--max-depth",
            "1",
            "--rollout-depth",
            "0",
            "--samples",
            "10",
            "--noise-std",
            "0.05",
            "--time-limit",
            "60",
            "--output",
            str(tmp_path),
            "--run-name",
            "live-cli",
        ]
    )
    assert code == 0
    assert (tmp_path / "live-cli" / "result.json").exists()

