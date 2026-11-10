"""Dataset containers, validation, and CSV loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


class DatasetError(ValueError):
    """Raised when a dataset is missing, malformed, or unusable."""


@dataclass(slots=True)
class Dataset:
    """A numeric dataset with named feature columns and one target column."""

    features: dict[str, np.ndarray]
    target: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.features:
            raise DatasetError("Dataset must contain at least one feature column")
        names = list(self.features)
        cleaned: dict[str, np.ndarray] = {}
        n_rows: int | None = None
        for name in names:
            array = np.asarray(self.features[name], dtype=float)
            if array.ndim != 1:
                raise DatasetError(f"Feature {name!r} must be one-dimensional")
            if not np.all(np.isfinite(array)):
                bad = int(np.flatnonzero(~np.isfinite(array))[0])
                raise DatasetError(f"Feature {name!r} contains a non-finite value at row {bad}")
            if n_rows is None:
                n_rows = int(array.size)
            elif int(array.size) != n_rows:
                raise DatasetError(
                    f"Feature {name!r} has {array.size} rows, expected {n_rows}"
                )
            cleaned[name] = array
        target = np.asarray(self.target, dtype=float)
        if target.ndim != 1:
            raise DatasetError("Target must be one-dimensional")
        if n_rows is None:
            n_rows = int(target.size)
        if int(target.size) != n_rows:
            raise DatasetError(
                f"Target has {target.size} rows but features have {n_rows} rows"
            )
        if n_rows < 2:
            raise DatasetError("Dataset must contain at least two rows")
        if not np.all(np.isfinite(target)):
            bad = int(np.flatnonzero(~np.isfinite(target))[0])
            raise DatasetError(f"Target contains a non-finite value at row {bad}")
        self.features = cleaned
        self.target = target

    @property
    def feature_names(self) -> list[str]:
        return list(self.features)

    @property
    def n_rows(self) -> int:
        return int(self.target.size)

    @classmethod
    def from_arrays(
        cls,
        features: Mapping[str, Iterable[float] | np.ndarray],
        target: Iterable[float] | np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Dataset":
        return cls(
            features={name: np.asarray(values, dtype=float) for name, values in features.items()},
            target=np.asarray(target, dtype=float),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "feature_names": self.feature_names,
            "metadata": dict(self.metadata),
        }


def _read_csv_rows(path: Path, delimiter: str, header: bool) -> tuple[list[str], list[list[str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            rows = list(reader)
    except FileNotFoundError as exc:
        raise DatasetError(f"Dataset file not found: {path}") from exc
    except OSError as exc:
        raise DatasetError(f"Could not read dataset file {path}: {exc}") from exc
    if not rows:
        raise DatasetError(f"Dataset file is empty: {path}")
    if header:
        names = [cell.strip() for cell in rows[0]]
        data = rows[1:]
    else:
        width = len(rows[0])
        names = [str(index) for index in range(width)]
        data = rows
    if not names or any(not name for name in names):
        raise DatasetError(f"Dataset file has an empty column name: {path}")
    if not data:
        raise DatasetError(f"Dataset file contains a header but no data rows: {path}")
    return names, data


def load_csv(
    path: str | Path,
    target: str,
    features: list[str] | None = None,
    delimiter: str = ",",
    header: bool = True,
) -> Dataset:
    """Load a numeric CSV dataset.

    ``features`` defaults to every column except ``target``. Missing columns,
    empty files, and non-numeric cells produce actionable errors.
    """

    csv_path = Path(path)
    names, rows = _read_csv_rows(csv_path, delimiter, header)
    if target not in names:
        raise DatasetError(
            f"Target column {target!r} not found. Available columns: {', '.join(names)}"
        )
    selected = list(features) if features else [name for name in names if name != target]
    if not selected:
        raise DatasetError("No feature columns selected")
    missing = [name for name in selected if name not in names]
    if missing:
        raise DatasetError(
            f"Feature column(s) not found: {', '.join(missing)}. "
            f"Available columns: {', '.join(names)}"
        )
    if target in selected:
        raise DatasetError("Target column must not also be used as a feature")
    width = len(names)
    columns: dict[str, list[float]] = {name: [] for name in [*selected, target]}
    for row_number, row in enumerate(rows, start=1):
        if len(row) != width:
            raise DatasetError(
                f"Row {row_number} has {len(row)} cells, expected {width}"
            )
        for name in [*selected, target]:
            index = names.index(name)
            cell = row[index].strip()
            if cell == "":
                raise DatasetError(
                    f"Column {name!r} contains a blank value at row {row_number}"
                )
            try:
                value = float(cell)
            except ValueError as exc:
                raise DatasetError(
                    f"Column {name!r} contains a non-numeric value at row {row_number}: {cell!r}"
                ) from exc
            if not np.isfinite(value):
                raise DatasetError(
                    f"Column {name!r} contains a non-finite value at row {row_number}"
                )
            columns[name].append(value)
    return Dataset.from_arrays(
        {name: columns[name] for name in selected},
        columns[target],
        metadata={
            "source": "csv",
            "path": str(csv_path),
            "target": target,
            "features": selected,
            "delimiter": delimiter,
            "header": header,
        },
    )
