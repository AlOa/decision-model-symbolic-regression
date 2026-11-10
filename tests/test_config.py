import pytest

from symbolic_regression.config import RunConfig, load_config_file, merge_config


def test_defaults_select_quadratic() -> None:
    config = merge_config()
    assert config.function == "quadratic"
    assert config.mode == "run"
    assert config.policy == "decision"


def test_cli_values_override_file_values() -> None:
    config = merge_config(
        {"function": "linear", "iterations": 5},
        {"iterations": 10, "seed": 7},
    )
    assert config.function == "linear"
    assert config.iterations == 10
    assert config.seed == 7


def test_unknown_config_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("function = 'linear'\nmystery = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_config_file(path)


def test_run_section_is_supported(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[run]\nfunction = 'sine'\niterations = 3\n", encoding="utf-8")
    values = load_config_file(path)
    assert values["function"] == "sine"
    assert values["iterations"] == 3


def test_missing_config_file_is_reported(tmp_path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_config_file(tmp_path / "missing.toml")


def test_invalid_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="samples"):
        RunConfig(function="linear", samples=1)
    with pytest.raises(ValueError, match="either"):
        RunConfig(function="linear", csv="data.csv")
    with pytest.raises(ValueError, match="Unknown policy"):
        RunConfig(function="linear", policy="mystery")
    with pytest.raises(ValueError, match="fit_threshold"):
        RunConfig(function="linear", fit_threshold=1.5)


def test_features_and_policies_are_split() -> None:
    config = merge_config(
        {"features": "x0, x1", "comparison_policies": "offline, uniform"}
    )
    assert config.features == ["x0", "x1"]
    assert config.comparison_policies == ("offline", "uniform")


def test_operators_are_split_and_validated() -> None:
    config = merge_config({"operators": "add, mul"})
    assert config.operators == ("add", "mul")
    assert config.to_dict()["operators"] == ["add", "mul"]
    with pytest.raises(ValueError, match="Unknown operators"):
        RunConfig(function="linear", operators=("mystery",))
    with pytest.raises(ValueError, match="at least one"):
        RunConfig(function="linear", operators=())


def test_to_dict_contains_search_and_data_options() -> None:
    config = RunConfig(function="linear", output="out")
    payload = config.to_dict()
    assert payload["function"] == "linear"
    assert payload["output"] == "out"
    assert payload["mode"] == "run"
