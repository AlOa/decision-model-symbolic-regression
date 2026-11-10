import json

import pytest

from symbolic_regression.cli import build_parser, main


def test_cli_help_lists_commands(capsys) -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "run" in help_text
    assert "compare" in help_text
    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])
    assert exc.value.code == 0
    run_help = capsys.readouterr().out
    assert "--function" in run_help
    assert "--noise-std" in run_help
    assert "--policy" in run_help


def test_run_command_offline(tmp_path, capsys) -> None:
    code = main(
        [
            "run",
            "--function",
            "linear",
            "--policy",
            "offline",
            "--iterations",
            "1",
            "--max-depth",
            "1",
            "--samples",
            "12",
            "--noise-std",
            "0",
            "--output",
            str(tmp_path),
            "--run-name",
            "cli-run",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Recovered expression" in output
    result_path = tmp_path / "cli-run" / "result.json"
    assert result_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["run"]["policy"] == "offline"
    assert payload["search"]["iterations"] == 1


def test_compare_command_offline(tmp_path, capsys) -> None:
    code = main(
        [
            "compare",
            "--function",
            "linear",
            "--policies",
            "offline,uniform",
            "--iterations",
            "1",
            "--max-depth",
            "1",
            "--samples",
            "10",
            "--noise-std",
            "0",
            "--output",
            str(tmp_path),
            "--run-name",
            "cli-compare",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Comparison written" in output
    assert (tmp_path / "cli-compare" / "comparison.json").exists()


def test_invalid_configuration_exits_non_zero(tmp_path, capsys) -> None:
    code = main(
        [
            "run",
            "--function",
            "linear",
            "--samples",
            "1",
            "--output",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert "samples" in capsys.readouterr().err


def test_unknown_config_key_exits_non_zero(tmp_path, capsys) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("function = 'linear'\nmystery = 1\n", encoding="utf-8")
    code = main(
        [
            "run",
            "--config",
            str(config_path),
            "--policy",
            "offline",
            "--output",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert "Unknown configuration key" in capsys.readouterr().err


def test_cli_overrides_config_file(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[run]\nfunction = 'linear'\niterations = 5\nsamples = 10\nnoise_std = 0.0\n",
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            "--config",
            str(config_path),
            "--policy",
            "offline",
            "--iterations",
            "1",
            "--max-depth",
            "1",
            "--output",
            str(tmp_path),
            "--run-name",
            "override",
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "override" / "result.json").read_text(encoding="utf-8"))
    assert payload["search"]["iterations"] == 1
    assert payload["run"]["function"] == "linear"


def test_run_with_operator_restriction(tmp_path) -> None:
    code = main(
        [
            "run",
            "--function",
            "quadratic",
            "--policy",
            "offline",
            "--operators",
            "mul",
            "--iterations",
            "3",
            "--max-depth",
            "2",
            "--max-steps",
            "2",
            "--samples",
            "12",
            "--noise-std",
            "0",
            "--output",
            str(tmp_path),
            "--run-name",
            "restricted",
        ]
    )
    assert code == 0
    payload = json.loads(
        (tmp_path / "restricted" / "result.json").read_text(encoding="utf-8")
    )
    assert payload["search"]["operators"] == ["mul"]


def test_run_csv_offline(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n0,0\n1,1\n2,4\n3,9\n", encoding="utf-8")
    code = main(
        [
            "run",
            "--csv",
            str(csv_path),
            "--target",
            "y",
            "--policy",
            "offline",
            "--iterations",
            "1",
            "--max-depth",
            "1",
            "--output",
            str(tmp_path),
            "--run-name",
            "csv-cli",
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "csv-cli" / "result.json").read_text(encoding="utf-8"))
    assert payload["run"]["csv"] == str(csv_path)
    assert payload["dataset"]["metadata"]["source"] == "csv"

