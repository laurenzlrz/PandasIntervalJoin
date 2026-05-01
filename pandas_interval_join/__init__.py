"""pandas-interval-join – efficient O(n + m) interval join for sorted DataFrames."""

from pandas_interval_join._core import (
    JoinHow,
    RowPredicate,
    interval_join,
    sorted_interval_join,
)

__all__ = [
    "interval_join",
    "sorted_interval_join",
    "JoinHow",
    "RowPredicate",
]

__version__ = "0.1.0"
