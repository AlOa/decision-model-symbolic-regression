"""Decision-model-guided symbolic regression with MCTS."""

from .api import (
    FitResult,
    SymbolicRegressor,
    available_operators,
    fit,
)
from .data import Dataset, DatasetError
from .expression import (
    BinaryOp,
    Constant,
    Expr,
    UnaryOp,
    Variable,
    evaluate,
    to_string,
)

__version__ = "0.1.0"

__all__ = [
    "BinaryOp",
    "Constant",
    "Dataset",
    "DatasetError",
    "Expr",
    "FitResult",
    "SymbolicRegressor",
    "UnaryOp",
    "Variable",
    "__version__",
    "available_operators",
    "evaluate",
    "fit",
    "to_string",
]
