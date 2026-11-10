"""Run configuration, TOML loading, and command-line merging."""

from __future__ import annotations

import tomllib
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .expression import OPERATORS

VALID_POLICIES = ("decision", "uniform", "offline")


@dataclass(slots=True)
class RunConfig:
    mode: str = "run"
    function: str | None = None
    csv: str | None = None
    target: str = "y"
    features: list[str] | None = None
    delimiter: str = ","
    header: bool = True
    model: str = "d3:4b"
    base_url: str = "http://localhost:11434"
    policy: str = "decision"
    comparison_policies: tuple[str, ...] = ("decision", "uniform")
    operators: tuple[str, ...] | None = None
    seed: int = 0
    samples: int = 64
    noise_std: float = 0.05
    noise_snr_db: float | None = None
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
    greedy: bool = False
    request_timeout: float = 30.0
    request_retries: int = 1
    output: str = "runs"
    run_name: str | None = None
    context_rows: int = 24
    residual_values: int = 16

    def __post_init__(self) -> None:
        self.mode = self.mode.lower()
        self.policy = self.policy.lower()
        if isinstance(self.features, str):
            self.features = _split_csv(self.features)
        elif self.features is not None:
            self.features = [str(name) for name in self.features]
        if isinstance(self.comparison_policies, str):
            self.comparison_policies = tuple(_split_csv(self.comparison_policies))
        else:
            self.comparison_policies = tuple(str(item).lower() for item in self.comparison_policies)
        if self.operators is not None:
            if isinstance(self.operators, str):
                self.operators = tuple(_split_csv(self.operators))
            else:
                self.operators = tuple(str(item) for item in self.operators)
            if not self.operators:
                raise ValueError("operators must contain at least one operator")
            unknown = [name for name in self.operators if name not in OPERATORS]
            if unknown:
                raise ValueError(f"Unknown operators: {', '.join(unknown)}")

        if self.mode not in ("run", "compare"):
            raise ValueError(f"Unknown mode {self.mode!r}; expected run or compare")
        if self.function is not None and self.csv is not None:
            raise ValueError("Provide either a suite function or a CSV dataset, not both")
        if self.function is None and self.csv is None:
            raise ValueError("Provide a suite function or a CSV dataset")
        if self.policy not in VALID_POLICIES:
            raise ValueError(
                f"Unknown policy {self.policy!r}; expected one of {', '.join(VALID_POLICIES)}"
            )
        invalid_policies = [item for item in self.comparison_policies if item not in VALID_POLICIES]
        if invalid_policies:
            raise ValueError(f"Unknown comparison policies: {', '.join(invalid_policies)}")
        if not self.comparison_policies:
            raise ValueError("At least one comparison policy is required")
        if not self.delimiter:
            raise ValueError("delimiter must not be empty")
        if self.samples < 2:
            raise ValueError("samples must be at least 2")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if self.max_depth < 1:
            raise ValueError("max_depth must be positive")
        if self.rollout_depth < 0:
            raise ValueError("rollout_depth must be non-negative")
        if self.c_puct < 0:
            raise ValueError("c_puct must be non-negative")
        if self.top_applications < 1:
            raise ValueError("top_applications must be positive")
        if self.max_operands < 1:
            raise ValueError("max_operands must be positive")
        if not (0.0 <= self.fit_threshold <= 1.0):
            raise ValueError("fit_threshold must be between 0 and 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if not isinstance(self.greedy, bool):
            raise ValueError("greedy must be a boolean")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if self.request_retries < 0:
            raise ValueError("request_retries must be non-negative")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if self.context_rows < 1:
            raise ValueError("context_rows must be positive")
        if self.residual_values < 1:
            raise ValueError("residual_values must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "function": self.function,
            "csv": self.csv,
            "target": self.target,
            "features": list(self.features) if self.features else None,
            "delimiter": self.delimiter,
            "header": self.header,
            "model": self.model,
            "base_url": self.base_url,
            "policy": self.policy,
            "comparison_policies": list(self.comparison_policies),
            "operators": list(self.operators) if self.operators else None,
            "seed": self.seed,
            "samples": self.samples,
            "noise_std": self.noise_std,
            "noise_snr_db": self.noise_snr_db,
            "iterations": self.iterations,
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
            "rollout_depth": self.rollout_depth,
            "c_puct": self.c_puct,
            "top_applications": self.top_applications,
            "max_operands": self.max_operands,
            "fit_threshold": self.fit_threshold,
            "max_steps": self.max_steps,
            "time_limit_seconds": self.time_limit_seconds,
            "greedy": self.greedy,
            "request_timeout": self.request_timeout,
            "request_retries": self.request_retries,
            "output": self.output,
            "run_name": self.run_name,
            "context_rows": self.context_rows,
            "residual_values": self.residual_values,
        }


CONFIG_KEYS = {field.name for field in fields(RunConfig)}


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a TOML configuration file and reject unknown keys."""

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"Configuration file not found: {config_path}") from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {config_path}: {exc}") from exc
    if "run" in data:
        if not isinstance(data["run"], dict):
            raise ValueError("The [run] section in the configuration must be a table")
        data = data["run"]
    unknown = set(data) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown configuration key(s): {', '.join(sorted(unknown))}")
    return dict(data)


def _defaults() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields(RunConfig):
        if field.default is not MISSING:
            result[field.name] = field.default
        elif field.default_factory is not MISSING:  # pragma: no cover - defensive
            result[field.name] = field.default_factory()
    return result


def merge_config(
    file_values: Mapping[str, Any] | None = None,
    cli_values: Mapping[str, Any] | None = None,
    mode: str = "run",
) -> RunConfig:
    """Merge defaults, file values, and CLI values, with CLI taking precedence."""

    values = _defaults()
    values["mode"] = mode
    values.update(file_values or {})
    values.update({key: value for key, value in (cli_values or {}).items() if value is not None})
    if values.get("function") is None and values.get("csv") is None:
        values["function"] = "quadratic"
    features = values.get("features")
    if isinstance(features, str):
        values["features"] = _split_csv(features) or None
    policies = values.get("comparison_policies")
    if isinstance(policies, str):
        values["comparison_policies"] = tuple(_split_csv(policies))
    operators = values.get("operators")
    if isinstance(operators, str):
        values["operators"] = tuple(_split_csv(operators)) or None
    return RunConfig(**values)
