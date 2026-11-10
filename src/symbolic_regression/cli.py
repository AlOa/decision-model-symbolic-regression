"""Command-line interface for symbolic-regression experiments."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .config import load_config_file, merge_config
from .data import DatasetError
from .decision.client import OllamaError
from .experiments.runner import (
    RunnerError,
    compare_experiments,
    format_outcome,
    run_experiment,
)


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", metavar="PATH", help="TOML configuration file")
    parser.add_argument("--function", help="built-in suite function name")
    parser.add_argument("--csv", help="CSV dataset path")
    parser.add_argument("--target", help="target column name")
    parser.add_argument("--features", help="comma-separated feature columns")
    parser.add_argument("--delimiter", help="CSV delimiter (default: ,)")
    parser.add_argument(
        "--header",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="CSV contains a header row (default: true)",
    )
    parser.add_argument("--model", help="Ollama decision model name")
    parser.add_argument("--base-url", dest="base_url", help="Ollama base URL")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--samples", type=int, help="number of generated samples")
    parser.add_argument("--noise-std", type=float, dest="noise_std", help="additive Gaussian noise std")
    parser.add_argument("--noise-snr-db", type=float, dest="noise_snr_db", help="target SNR in dB")
    parser.add_argument("--iterations", type=int, help="maximum MCTS iterations")
    parser.add_argument("--max-nodes", type=int, dest="max_nodes", help="maximum tree nodes")
    parser.add_argument("--max-depth", type=int, dest="max_depth", help="maximum expression depth")
    parser.add_argument("--rollout-depth", type=int, dest="rollout_depth", help="rollout steps")
    parser.add_argument("--c-puct", type=float, dest="c_puct", help="PUCT exploration constant")
    parser.add_argument(
        "--top-applications",
        type=int,
        dest="top_applications",
        help="operand applications kept per operator",
    )
    parser.add_argument("--max-operands", type=int, dest="max_operands", help="operand bank size")
    parser.add_argument(
        "--fit-threshold", type=float, dest="fit_threshold", help="R-squared stop threshold"
    )
    parser.add_argument("--max-steps", type=int, dest="max_steps", help="maximum construction steps")
    parser.add_argument(
        "--time-limit",
        type=float,
        dest="time_limit_seconds",
        help="wall-clock search budget in seconds",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        default=None,
        help="follow the highest-probability operator path instead of branching",
    )
    parser.add_argument(
        "--request-timeout", type=float, dest="request_timeout", help="Ollama request timeout"
    )
    parser.add_argument(
        "--request-retries", type=int, dest="request_retries", help="Ollama request retries"
    )
    parser.add_argument("--output", help="output directory (default: runs)")
    parser.add_argument("--run-name", dest="run_name", help="output subdirectory name")
    parser.add_argument("--context-rows", type=int, dest="context_rows", help="rows sent to the model")
    parser.add_argument(
        "--residual-values",
        type=int,
        dest="residual_values",
        help="residual values sent to the model",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symbolic-regression",
        description="Decision-model-guided symbolic regression with MCTS.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    run_parser = subparsers.add_parser(
        "run", parents=[common], help="run one experiment"
    )
    run_parser.add_argument(
        "--policy",
        choices=("decision", "uniform", "offline"),
        help="operator policy (default: decision)",
    )
    compare_parser = subparsers.add_parser(
        "compare", parents=[common], help="compare policies with the same seed and budget"
    )
    compare_parser.add_argument(
        "--policies",
        help="comma-separated policies to compare (default: decision,uniform)",
    )
    return parser


_CLI_FIELDS = (
    "function",
    "csv",
    "target",
    "features",
    "delimiter",
    "header",
    "model",
    "base_url",
    "seed",
    "samples",
    "noise_std",
    "noise_snr_db",
    "iterations",
    "max_nodes",
    "max_depth",
    "rollout_depth",
    "c_puct",
    "top_applications",
    "max_operands",
    "fit_threshold",
    "max_steps",
    "time_limit_seconds",
    "greedy",
    "request_timeout",
    "request_retries",
    "output",
    "run_name",
    "context_rows",
    "residual_values",
)


def _cli_values(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in _CLI_FIELDS:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    if args.command == "run":
        policy = getattr(args, "policy", None)
        if policy is not None:
            values["policy"] = policy
    else:
        policies = getattr(args, "policies", None)
        if policies is not None:
            values["comparison_policies"] = policies
    return values


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        file_values = load_config_file(args.config) if args.config else {}
        config = merge_config(file_values, _cli_values(args), mode=args.command)
        if args.command == "run":
            outcome = run_experiment(config)
            print(format_outcome(outcome))
        else:
            comparison = compare_experiments(config)
            print(f"Comparison written to {comparison.comparison_path}")
            for outcome in comparison.outcomes:
                print()
                print(format_outcome(outcome))
        return 0
    except (
        ValueError,
        TypeError,
        DatasetError,
        OllamaError,
        RunnerError,
        FileNotFoundError,
        KeyError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
