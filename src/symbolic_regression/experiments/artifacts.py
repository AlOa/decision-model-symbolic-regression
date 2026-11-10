"""Result, trace, and comparison artifact writing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..config import RunConfig
from ..data import Dataset
from ..expression import node_count, to_string
from ..result import SearchResult


@dataclass(slots=True)
class ArtifactPaths:
    directory: Path
    result: Path
    trace: Path
    summary: Path


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or "run"


def dataset_name(dataset: Dataset) -> str:
    if dataset.metadata.get("function"):
        return str(dataset.metadata["function"])
    path = dataset.metadata.get("path")
    return Path(path).stem if path else "dataset"


def run_directory(config: RunConfig, dataset: Dataset) -> Path:
    name = config.run_name or f"{dataset_name(dataset)}-{config.policy}-seed{config.seed}"
    return Path(config.output) / _slug(name)


def _result_payload(
    config: RunConfig,
    dataset: Dataset,
    result: SearchResult,
    policy_name: str,
) -> dict[str, Any]:
    result_dict = result.to_dict()
    result_dict["complexity"] = (
        node_count(result.best_expression) if result.best_expression is not None else None
    )
    result_dict["ground_truth"] = dataset.metadata.get("expression")
    return {
        "run": {
            "mode": config.mode,
            "policy": policy_name,
            "model": config.model if policy_name == "decision" else None,
            "seed": config.seed,
            "function": dataset.metadata.get("function"),
            "csv": dataset.metadata.get("path"),
            "output": str(config.output),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": result.elapsed_seconds,
        },
        "search": {
            "iterations": config.iterations,
            "max_nodes": config.max_nodes,
            "max_depth": config.max_depth,
            "rollout_depth": config.rollout_depth,
            "c_puct": config.c_puct,
            "top_applications": config.top_applications,
            "max_operands": config.max_operands,
            "fit_threshold": config.fit_threshold,
            "max_steps": config.max_steps,
            "time_limit_seconds": config.time_limit_seconds,
            "greedy": config.greedy,
        },
        "dataset": dataset.to_dict(),
        "result": result_dict,
    }


def _summary(
    config: RunConfig,
    dataset: Dataset,
    result: SearchResult,
    policy_name: str,
) -> str:
    lines = [
        "# Run summary",
        "",
        f"- Function: {dataset.metadata.get('function', 'CSV dataset')}",
        f"- Policy: {policy_name}",
        f"- Recovered expression: {to_string(result.best_expression) if result.best_expression else 'none'}",
        f"- Ground truth: {dataset.metadata.get('expression', 'n/a')}",
    ]
    if result.best_evaluation is not None and result.best_evaluation.valid:
        evaluation = result.best_evaluation
        lines.extend(
            [
                f"- MSE: {evaluation.mse:.6g}",
                f"- NMSE: {evaluation.nmse:.6g}",
                f"- R-squared: {evaluation.r2:.6g}",
            ]
        )
    else:
        lines.append("- Metrics: no feasible expression")
    lines.extend(
        [
            f"- Complexity: {node_count(result.best_expression) if result.best_expression else 0}",
            f"- Decision calls: {result.decision_calls}",
            f"- Fallback calls: {result.fallback_calls}",
            f"- Iterations: {result.iterations}",
            f"- Stop reason: {result.stop_reason.value}",
            f"- Elapsed: {result.elapsed_seconds:.3f}s",
            f"- Seed: {config.seed}",
            f"- Noise: {dataset.metadata.get('noise_mode', 'n/a')}={dataset.metadata.get('noise_value', 'n/a')}",
            "",
        ]
    )
    return "\n".join(lines)


def write_run_artifacts(
    config: RunConfig,
    dataset: Dataset,
    result: SearchResult,
    policy_name: str,
    directory: Path,
) -> ArtifactPaths:
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    trace_path = directory / "trace.json"
    summary_path = directory / "summary.md"
    result_path.write_text(
        json.dumps(
            _result_payload(config, dataset, result, policy_name),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    trace_path.write_text(
        json.dumps(
            {
                "decisions": result.decision_records,
                "iterations": result.history,
                "stop_reason": result.stop_reason.value,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(_summary(config, dataset, result, policy_name), encoding="utf-8")
    return ArtifactPaths(directory, result_path, trace_path, summary_path)


def write_comparison_artifacts(
    config: RunConfig,
    dataset: Dataset,
    outcomes: Iterable[tuple[str, SearchResult]],
    directory: Path,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    collected = list(outcomes)
    payload = {
        "run": {
            "mode": "compare",
            "seed": config.seed,
            "function": dataset.metadata.get("function"),
            "csv": dataset.metadata.get("path"),
            "policies": [policy for policy, _ in collected],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "dataset": dataset.to_dict(),
        "results": {},
    }
    for policy, result in collected:
        result_dict = result.to_dict()
        result_dict["complexity"] = (
            node_count(result.best_expression) if result.best_expression is not None else None
        )
        result_dict["ground_truth"] = dataset.metadata.get("expression")
        payload["results"][policy] = result_dict
    comparison_path = directory / "comparison.json"
    comparison_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Policy comparison",
        "",
        f"Seed: {config.seed}",
        f"Function: {dataset.metadata.get('function', 'CSV dataset')}",
        "",
        "| Policy | Expression | R-squared | NMSE | MSE | Complexity | Decisions | Stop reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for policy, result in collected:
        evaluation = result.best_evaluation
        expression = to_string(result.best_expression) if result.best_expression else "none"
        r2 = f"{evaluation.r2:.6g}" if evaluation and evaluation.valid else "n/a"
        nmse = f"{evaluation.nmse:.6g}" if evaluation and evaluation.valid else "n/a"
        mse = f"{evaluation.mse:.6g}" if evaluation and evaluation.valid else "n/a"
        complexity = node_count(result.best_expression) if result.best_expression else 0
        lines.append(
            f"| {policy} | {expression} | {r2} | {nmse} | {mse} | {complexity} | "
            f"{result.decision_calls} | {result.stop_reason.value} |"
        )
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison_path
