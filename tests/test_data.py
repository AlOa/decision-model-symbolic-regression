import numpy as np
import pytest

from symbolic_regression.data import Dataset, DatasetError, load_csv


def test_valid_dataset_is_accepted() -> None:
    dataset = Dataset.from_arrays({"x": [0, 1, 2]}, [0, 1, 4])
    assert dataset.n_rows == 3
    assert dataset.feature_names == ["x"]


def test_non_numeric_target_is_rejected(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("x,y\n0,1\n1,oops\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="non-numeric"):
        load_csv(path, target="y")


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(DatasetError, match="rows"):
        Dataset(features={"x": np.array([1.0, 2.0])}, target=np.array([1.0, 2.0, 3.0]))


def test_missing_csv_column_is_reported(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("x,y\n0,1\n1,2\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="missing"):
        load_csv(path, target="y", features=["x", "missing"])


def test_empty_csv_is_rejected(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match="empty"):
        load_csv(path, target="y")


def test_valid_csv_is_loaded(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("x,y\n0,1\n1,2\n2,5\n", encoding="utf-8")
    dataset = load_csv(path, target="y")
    assert dataset.feature_names == ["x"]
    assert dataset.n_rows == 3
    np.testing.assert_allclose(dataset.target, [1.0, 2.0, 5.0])
