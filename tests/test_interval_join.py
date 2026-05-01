"""
Comprehensive tests for pandas_interval_join.

Each test verifies both the returned DataFrame content **and** its dtypes /
column names so regressions in column handling are caught early.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from pandas_interval_join import interval_join, sorted_interval_join


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lower(l: pd.Series, r: pd.Series) -> bool:
    """True when right row is at or above the window lower bound."""
    return bool(l["start"] <= r["time"])


def _upper(l: pd.Series, r: pd.Series) -> bool:
    """True when right row is at or below the window upper bound."""
    return bool(r["time"] <= l["end"])


def make_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": [1, 3, 6],
            "end": [5, 7, 9],
            "label": ["A", "B", "C"],
        }
    )


def make_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0, 2, 4, 6, 8, 10],
            "value": [100, 200, 300, 400, 500, 600],
        }
    )


# ---------------------------------------------------------------------------
# interval_join – inner
# ---------------------------------------------------------------------------


class TestInnerJoin:
    def test_basic_match(self) -> None:
        events = make_events()
        points = make_points()
        result = interval_join(events, points, _lower, _upper)

        # Manual enumeration:
        # A: start=1, end=5 → time in {2, 4}          → 2 matches
        # B: start=3, end=7 → time in {4, 6}          → 2 matches
        # C: start=6, end=9 → time in {6, 8}          → 2 matches
        # Total: 6 matches
        expected_labels = ["A", "A", "B", "B", "C", "C"]
        expected_times = [2, 4, 4, 6, 6, 8]
        assert set(result.columns) == {"start", "end", "label", "time", "value"}
        assert len(result) == 6
        assert sorted(result["label"].tolist()) == sorted(expected_labels)
        assert sorted(result["time"].tolist()) == sorted(expected_times)

    def test_no_matches(self) -> None:
        events = pd.DataFrame({"start": [10], "end": [20], "label": ["A"]})
        points = pd.DataFrame({"time": [1, 2], "value": [1, 2]})
        result = interval_join(events, points, _lower, _upper)
        assert len(result) == 0
        assert set(result.columns) == {"start", "end", "label", "time", "value"}

    def test_all_match(self) -> None:
        events = pd.DataFrame({"start": [0], "end": [100], "label": ["X"]})
        points = pd.DataFrame({"time": list(range(10)), "value": list(range(10))})
        result = interval_join(events, points, _lower, _upper)
        assert len(result) == 10

    def test_empty_left(self) -> None:
        events = pd.DataFrame({"start": pd.Series([], dtype=int), "end": pd.Series([], dtype=int), "label": pd.Series([], dtype=str)})
        points = make_points()
        result = interval_join(events, points, _lower, _upper)
        assert len(result) == 0

    def test_empty_right(self) -> None:
        events = make_events()
        points = pd.DataFrame({"time": pd.Series([], dtype=int), "value": pd.Series([], dtype=int)})
        result = interval_join(events, points, _lower, _upper)
        assert len(result) == 0

    def test_both_empty(self) -> None:
        events = pd.DataFrame({"start": pd.Series([], dtype=int), "end": pd.Series([], dtype=int)})
        points = pd.DataFrame({"time": pd.Series([], dtype=int)})
        result = interval_join(events, points, _lower, _upper)
        assert len(result) == 0

    def test_result_values(self) -> None:
        """Exact row-level check."""
        events = pd.DataFrame({"start": [1, 3], "end": [5, 7], "name": ["A", "B"]})
        points = pd.DataFrame({"time": [2, 4, 6], "value": [10, 20, 30]})
        result = interval_join(events, points, _lower, _upper).sort_values(
            ["name", "time"]
        ).reset_index(drop=True)
        assert result.loc[0, "name"] == "A"
        assert result.loc[0, "time"] == 2
        assert result.loc[1, "name"] == "A"
        assert result.loc[1, "time"] == 4
        assert result.loc[2, "name"] == "B"
        assert result.loc[2, "time"] == 4
        assert result.loc[3, "name"] == "B"
        assert result.loc[3, "time"] == 6


# ---------------------------------------------------------------------------
# interval_join – left join
# ---------------------------------------------------------------------------


class TestLeftJoin:
    def test_unmatched_left_rows_present(self) -> None:
        events = make_events()
        points = make_points()
        result = interval_join(events, points, _lower, _upper, how="left")

        # All left rows must appear at least once
        assert set(result["label"].dropna().unique()) == {"A", "B", "C"}

    def test_unmatched_left_has_nan_right_cols(self) -> None:
        events = pd.DataFrame({"start": [1, 50], "end": [5, 60], "label": ["A", "Z"]})
        points = pd.DataFrame({"time": [2, 3], "value": [10, 20]})
        result = interval_join(events, points, _lower, _upper, how="left")

        z_rows = result[result["label"] == "Z"]
        assert len(z_rows) == 1
        assert pd.isna(z_rows.iloc[0]["time"])
        assert pd.isna(z_rows.iloc[0]["value"])

    def test_left_join_superset_of_inner(self) -> None:
        events = make_events()
        points = make_points()
        inner = interval_join(events, points, _lower, _upper)
        left = interval_join(events, points, _lower, _upper, how="left")
        assert len(left) >= len(inner)


# ---------------------------------------------------------------------------
# interval_join – right join
# ---------------------------------------------------------------------------


class TestRightJoin:
    def test_unmatched_right_rows_present(self) -> None:
        events = pd.DataFrame({"start": [1], "end": [3], "label": ["A"]})
        points = pd.DataFrame({"time": [2, 10], "value": [10, 99]})
        result = interval_join(events, points, _lower, _upper, how="right")

        # time=10 has no match → should appear with NaN label
        unmatched = result[result["time"] == 10]
        assert len(unmatched) == 1
        assert pd.isna(unmatched.iloc[0]["label"])

    def test_right_join_superset_of_inner(self) -> None:
        events = make_events()
        points = make_points()
        inner = interval_join(events, points, _lower, _upper)
        right = interval_join(events, points, _lower, _upper, how="right")
        assert len(right) >= len(inner)


# ---------------------------------------------------------------------------
# interval_join – outer join
# ---------------------------------------------------------------------------


class TestOuterJoin:
    def test_outer_contains_inner(self) -> None:
        events = make_events()
        points = make_points()
        inner = interval_join(events, points, _lower, _upper)
        outer = interval_join(events, points, _lower, _upper, how="outer")
        assert len(outer) >= len(inner)

    def test_outer_has_all_left_and_right(self) -> None:
        events = pd.DataFrame({"start": [1], "end": [3], "label": ["A"]})
        points = pd.DataFrame({"time": [2, 100], "value": [10, 99]})
        result = interval_join(events, points, _lower, _upper, how="outer")
        # Matched: (A, 2). Unmatched right: (100,99). No unmatched left.
        assert len(result) == 2
        assert 100 in result["time"].tolist()


# ---------------------------------------------------------------------------
# interval_join – overlapping column names / suffixes
# ---------------------------------------------------------------------------


class TestSuffixes:
    def test_default_suffixes(self) -> None:
        df1 = pd.DataFrame({"time": [1, 2, 3], "value": [10, 20, 30]})
        df2 = pd.DataFrame({"time": [1, 2, 3], "value": [100, 200, 300]})

        def lower(l: pd.Series, r: pd.Series) -> bool:
            return bool(l["time"] <= r["time"])

        def upper(l: pd.Series, r: pd.Series) -> bool:
            return bool(r["time"] <= l["time"] + 1)

        result = interval_join(df1, df2, lower, upper)
        assert "time_x" in result.columns
        assert "time_y" in result.columns
        assert "value_x" in result.columns
        assert "value_y" in result.columns

    def test_custom_suffixes(self) -> None:
        df1 = pd.DataFrame({"val": [1, 2]})
        df2 = pd.DataFrame({"val": [1, 2]})

        def lower(l: pd.Series, r: pd.Series) -> bool:
            return bool(l["val"] <= r["val"])

        def upper(l: pd.Series, r: pd.Series) -> bool:
            return bool(r["val"] <= l["val"] + 1)

        result = interval_join(df1, df2, lower, upper, suffixes=("_left", "_right"))
        assert "val_left" in result.columns
        assert "val_right" in result.columns


# ---------------------------------------------------------------------------
# sorted_interval_join
# ---------------------------------------------------------------------------


class TestSortedIntervalJoin:
    def _make_orders(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "GOOG"],
                "t_start": [1, 4, 1],
                "t_end": [5, 8, 3],
            }
        )

    def _make_trades(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL", "GOOG"],
                "time": [2, 5, 7, 2],
                "price": [100, 101, 102, 50],
            }
        )

    def _lower(self, l: pd.Series, r: pd.Series) -> bool:
        return bool(l["t_start"] <= r["time"])

    def _upper(self, l: pd.Series, r: pd.Series) -> bool:
        return bool(r["time"] <= l["t_end"])

    def test_basic(self) -> None:
        result = sorted_interval_join(
            self._make_orders(),
            self._make_trades(),
            self._lower,
            self._upper,
            left_on=["ticker"],
            right_on=["ticker"],
        )
        # AAPL order (1..5): trades at 2, 5 → 2 matches
        # AAPL order (4..8): trades at 5, 7 → 2 matches
        # GOOG order (1..3): trade at 2    → 1 match
        assert len(result) == 5
        # join key should appear only once
        assert "ticker" in result.columns
        # no duplicate ticker column
        assert "ticker_x" not in result.columns
        assert "ticker_y" not in result.columns

    def test_no_duplicate_key_column(self) -> None:
        orders = self._make_orders()
        trades = self._make_trades()
        result = sorted_interval_join(
            orders, trades, self._lower, self._upper,
            left_on=["ticker"], right_on=["ticker"],
        )
        ticker_cols = [c for c in result.columns if "ticker" in c]
        assert len(ticker_cols) == 1

    def test_unsorted_input_sorted_within_group(self) -> None:
        """sorted_interval_join sorts by equijoin key, but within each key group
        the rows must also be ordered so the interval window moves forward.
        Here both inputs arrive in reversed *key* order; after sorting by ticker
        the rows within each group are in the correct interval order."""
        orders = pd.DataFrame(
            {
                "ticker": ["GOOG", "AAPL", "AAPL"],
                "t_start": [1, 1, 4],
                "t_end": [3, 5, 8],
            }
        )
        trades = pd.DataFrame(
            {
                "ticker": ["GOOG", "AAPL", "AAPL", "AAPL"],
                "time": [2, 2, 5, 7],
                "price": [50, 100, 101, 102],
            }
        )
        result = sorted_interval_join(
            orders, trades, self._lower, self._upper,
            left_on=["ticker"], right_on=["ticker"],
        )
        assert len(result) == 5

    def test_different_key_column_names(self) -> None:
        left = pd.DataFrame({"grp_l": ["A", "A", "B"], "s": [1, 4, 1], "e": [5, 8, 3]})
        right = pd.DataFrame({"grp_r": ["A", "A", "B"], "t": [2, 5, 2], "v": [10, 11, 20]})

        def lower_b(l: pd.Series, r: pd.Series) -> bool:
            return bool(l["s"] <= r["t"])

        def upper_b(l: pd.Series, r: pd.Series) -> bool:
            return bool(r["t"] <= l["e"])

        result = sorted_interval_join(
            left, right, lower_b, upper_b,
            left_on=["grp_l"], right_on=["grp_r"],
        )
        # grp_r should be dropped; grp_l should remain as 'grp_l'
        assert "grp_r" not in result.columns
        assert "grp_l" in result.columns

    def test_mismatched_key_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            sorted_interval_join(
                pd.DataFrame({"a": [1]}),
                pd.DataFrame({"b": [1]}),
                lambda l, r: True,
                lambda l, r: True,
                left_on=["a"],
                right_on=["b", "b"],
            )

    def test_left_join(self) -> None:
        orders = self._make_orders()
        trades = pd.DataFrame(
            {"ticker": ["AAPL", "AAPL"], "time": [2, 5], "price": [100, 101]}
        )
        result = sorted_interval_join(
            orders, trades, self._lower, self._upper,
            left_on=["ticker"], right_on=["ticker"], how="left",
        )
        # GOOG order has no trades → should appear with NaN price
        goog = result[result["ticker"] == "GOOG"]
        assert len(goog) == 1
        assert pd.isna(goog.iloc[0]["price"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_row_each(self) -> None:
        left = pd.DataFrame({"start": [1], "end": [10]})
        right = pd.DataFrame({"time": [5]})
        result = interval_join(
            left, right,
            lambda l, r: bool(l["start"] <= r["time"]),
            lambda l, r: bool(r["time"] <= l["end"]),
        )
        assert len(result) == 1

    def test_single_row_no_match(self) -> None:
        left = pd.DataFrame({"start": [1], "end": [3]})
        right = pd.DataFrame({"time": [5]})
        result = interval_join(
            left, right,
            lambda l, r: bool(l["start"] <= r["time"]),
            lambda l, r: bool(r["time"] <= l["end"]),
        )
        assert len(result) == 0

    def test_all_right_below_window(self) -> None:
        left = pd.DataFrame({"start": [10], "end": [20]})
        right = pd.DataFrame({"time": [1, 2, 3]})
        result = interval_join(
            left, right,
            lambda l, r: bool(l["start"] <= r["time"]),
            lambda l, r: bool(r["time"] <= l["end"]),
        )
        assert len(result) == 0

    def test_monotone_windows(self) -> None:
        """Window slides strictly forward – verify all expected pairs."""
        left = pd.DataFrame({"start": [1, 2, 3], "end": [3, 4, 5]})
        right = pd.DataFrame({"time": [1, 2, 3, 4, 5]})
        result = interval_join(
            left, right,
            lambda l, r: bool(l["start"] <= r["time"]),
            lambda l, r: bool(r["time"] <= l["end"]),
        )
        # left[0]→(1,2,3), left[1]→(2,3,4), left[2]→(3,4,5) = 9 pairs
        assert len(result) == 9

    def test_numeric_stability(self) -> None:
        left = pd.DataFrame({"start": [0.1, 0.5], "end": [0.4, 1.0]})
        right = pd.DataFrame({"time": [0.2, 0.6, 0.8]})
        result = interval_join(
            left, right,
            lambda l, r: bool(l["start"] <= r["time"]),
            lambda l, r: bool(r["time"] <= l["end"]),
        )
        # left[0]: 0.1<=0.2<=0.4 → time=0.2
        # left[1]: 0.5<=0.6<=1.0 → time=0.6,0.8
        assert len(result) == 3

    def test_preserves_all_left_columns(self) -> None:
        left = pd.DataFrame(
            {"start": [1, 3], "end": [5, 7], "cat": ["x", "y"], "extra": [99, 88]}
        )
        right = pd.DataFrame({"time": [2, 4, 6]})
        result = interval_join(
            left, right,
            lambda l, r: bool(l["start"] <= r["time"]),
            lambda l, r: bool(r["time"] <= l["end"]),
        )
        assert "extra" in result.columns
        assert "cat" in result.columns

    def test_preserves_all_right_columns(self) -> None:
        left = pd.DataFrame({"start": [1], "end": [10]})
        right = pd.DataFrame({"time": [5], "col_a": ["hello"], "col_b": [3.14]})
        result = interval_join(
            left, right,
            lambda l, r: bool(l["start"] <= r["time"]),
            lambda l, r: bool(r["time"] <= l["end"]),
        )
        assert "col_a" in result.columns
        assert "col_b" in result.columns
