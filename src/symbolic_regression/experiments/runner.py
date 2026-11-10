"""Experiment orchestration for single runs and policy comparisons."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ..config import RunConfig
from ..data import Dataset, load_csv
from ..decision.client import OllamaClient, OllamaError
from ..decision.policy import (
    OfflineOperatorPolicy,
    OllamaOperatorPolicy,
    OperatorPolicy,
    UniformOperatorPolicy,
)
from ..expression import Variable, to_string
from ..result import SearchResult
from ..search.mcts import MCTS, MCTSConfig
from .artifacts import (
    ArtifactPaths,
    dataset_name,
    run_directory,
    write_comparison_artifacts,
    write_run_artifacts,
)
from .datasets import generate_dataset


class RunnerError(RuntimeError):
    """Raised when an experiment cannot be started or completed."""


@dataclass(slots=True)
class RunOutcome:
    config: RunConfig
    dataset: Dataset
    result: SearchResult
    policy_name: str
    artifacts: ArtifactPaths


@dataclass(slots=True)
class ComparisonOutcome:
    config: RunConfig
    dataset: Dataset
    outcomes: list[RunOutcome]
    directory: Path
    comparison_path: Path


def load_dataset(config: RunConfig) -> Dataset:
    if config.csv is not None:
        return load_csv(
            config.csv,
            target=config.target,
            features=config.features,
            delimiter=config.delimiter,
            header=config.header,
        )
    if config.function is None:
        raise RunnerError("No suite function or CSV dataset configured")
    return generate_dataset(
        config.function,
        n_samples=config.samples,
        seed=config.seed,
        noise_std=config.noise_std,
        noise_snr_db=config.noise_snr_db,
    )


def make_policy(config: RunConfig) -> tuple[OperatorPolicy, OllamaClient | None]:
    if config.policy == "decision":
        client = OllamaClient(
            base_url=config.base_url,
            timeout=config.request_timeout,
            retries=config.request_retries,
        )
        try:
            client.readiness(config.model)
        except OllamaError as exc:
            client.close()
            raise RunnerError(
                f"Decision model {config.model!r} is not ready: {exc}. "
                "Start Ollama and use a model with the 'decision' capability, "
                "or run with --policy offline."
            ) from exc
        return OllamaOperatorPolicy(client=client, model=config.model), client
    if config.policy == "uniform":
        return UniformOperatorPolicy(), None
    if config.policy == "offline":
        return OfflineOperatorPolicy(seed=config.seed), None
    raise RunnerError(f"Unknown policy {config.policy!r}")


def make_mcts_config(config: RunConfig) -> MCTSConfig:
    return MCTSConfig(
        iterations=config.iterations,
        max_nodes=config.max_nodes,
        max_depth=config.max_depth,
        rollout_depth=config.rollout_depth,
        c_puct=config.c_puct,
        top_applications=config.top_applications,
        max_operands=config.max_operands,
        fit_threshold=config.fit_threshold,
        max_steps=config.max_steps,
        time_limit_seconds=config.time_limit_seconds,
        seed=config.seed,
        greedy=config.greedy,
    )


def run_experiment(
    config: RunConfig, output_dir: Path | None = None
) -> RunOutcome:
    dataset = load_dataset(config)
    policy, client = make_policy(config)
    try:
        search = MCTS(
            initial_expression=Variable(dataset.feature_names[0]),
            dataset=dataset,
            policy=policy,
            config=make_mcts_config(config),
        )
        result = search.run()
    finally:
        if client is not None:
            client.close()
    directory = output_dir if output_dir is not None else run_directory(config, dataset)
    artifacts = write_run_artifacts(
        config, dataset, result, policy.name, directory
    )
    return RunOutcome(
        config=config,
        dataset=dataset,
        result=result,
        policy_name=policy.name,
        artifacts=artifacts,
    )


def compare_experiments(config: RunConfig) -> ComparisonOutcome:
    dataset = load_dataset(config)
    directory = Path(config.output) / (
        config.run_name or f"compare-{dataset_name(dataset)}-seed{config.seed}"
    )
    outcomes: list[RunOutcome] = []
    for policy_name in config.comparison_policies:
        subconfig = replace(
            config,
            mode="run",
            policy=policy_name,
            run_name=None,
        )
        outcomes.append(run_experiment(subconfig, output_dir=directory / policy_name))
    comparison_path = write_comparison_artifacts(
        config,
        dataset,
        [(outcome.policy_name, outcome.result) for outcome in outcomes],
        directory,
    )
    return ComparisonOutcome(
        config=config,
        dataset=dataset,
        outcomes=outcomes,
        directory=directory,
        comparison_path=comparison_path,
    )


def format_outcome(outcome: RunOutcome) -> str:
    result = outcome.result
    evaluation = result.best_evaluation
    lines = [
        f"Policy: {outcome.policy_name}",
        f"Recovered expression: {to_string(result.best_expression) if result.best_expression else 'none'}",
    ]
    if evaluation is not None and evaluation.valid:
        lines.append(f"MSE: {evaluation.mse:.6g}")
        lines.append(f"NMSE: {evaluation.nmse:.6g}")
        lines.append(f"R-squared: {evaluation.r2:.6g}")
    else:
        lines.append("Metrics: no feasible expression")
    lines.append(f"Stop reason: {result.stop_reason.value}")
    lines.append(f"Decision calls: {result.decision_calls} (fallbacks: {result.fallback_calls})")
    lines.append(f"Artifacts: {outcome.artifacts.directory}")
    return "\n".join(lines)
