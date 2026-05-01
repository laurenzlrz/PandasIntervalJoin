"""
Core two-pointer interval join implementation.

Terminology
-----------
Given two pre-sorted DataFrames *left* and *right*, an **interval join** finds
every pair of rows ``(left[i], right[k])`` where

    lower_bound(left[i], right[k]) is True   AND
    upper_bound(left[i], right[k]) is True

The key requirement is the **monotone window property**: as *i* increases,
the contiguous range of *right* indices that satisfy both conditions can only
move forward (never backward).  When this property holds the algorithm runs in
O(n + m) comparisons, where *n* = ``len(left)`` and *m* = ``len(right)``.

Condition semantics
-------------------
``lower_bound(left_row, right_row) -> bool``
    Returns ``True`` when *right_row* is **at or above** the lower boundary of
    *left_row*'s match window.  ``False`` means *right_row* is below the
    window; the right pointer advances past it.

``upper_bound(left_row, right_row) -> bool``
    Returns ``True`` when *right_row* is **at or below** the upper boundary of
    *left_row*'s match window.  ``False`` means *right_row* is above the
    window; the inner scan stops.

Example – join on ``left.start <= right.time <= left.end``::

    lower_bound = lambda l, r: l["start"] <= r["time"]
    upper_bound = lambda l, r: r["time"] <= l["end"]
"""

from __future__ import annotations

from typing import Callable, List, Literal, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

#: Join predicate: ``(left_row, right_row) -> bool``.
RowPredicate = Callable[[pd.Series, pd.Series], bool]

#: Allowed values for the *how* parameter.
JoinHow = Literal["inner", "left", "right", "outer"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _two_pointer_match(
    left: pd.DataFrame,
    right: pd.DataFrame,
    lower_bound: RowPredicate,
    upper_bound: RowPredicate,
) -> Tuple[List[int], List[int]]:
    """Return matching (left_position, right_position) pairs.

    Parameters
    ----------
    left:
        Left DataFrame, pre-sorted and zero-indexed.
    right:
        Right DataFrame, pre-sorted and zero-indexed.
    lower_bound:
        See module docstring.
    upper_bound:
        See module docstring.

    Returns
    -------
    tuple[list[int], list[int]]
        Two equal-length lists ``(left_positions, right_positions)`` where
        each pair ``(left_positions[p], right_positions[p])`` is a match.
    """
    left_positions: List[int] = []
    right_positions: List[int] = []

    j: int = 0  # rolling start of the right-side window
    n_right: int = len(right)

    for i in range(len(left)):
        left_row: pd.Series = left.iloc[i]

        # Advance j past right rows that are *below* the window for left[i].
        # lower_bound == False  ⟹  right[j] is below window  ⟹  skip it.
        while j < n_right and not lower_bound(left_row, right.iloc[j]):
            j += 1

        if j >= n_right:
            # All remaining right rows are below every remaining left row.
            break

        # Scan right rows inside the window.
        # upper_bound == False  ⟹  right[k] is above window  ⟹  stop.
        previous_j = j
        k = j
        while k < n_right and upper_bound(left_row, right.iloc[k]):
            left_positions.append(i)
            right_positions.append(k)
            k += 1

        # Restore j so the next left row can start its own window scan from
        # previous_j (the window start can only move forward, never backward).
        j = previous_j

    return left_positions, right_positions


def _build_joined(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_positions: List[int],
    right_positions: List[int],
    how: JoinHow,
    suffixes: Tuple[str, str],
) -> pd.DataFrame:
    """Assemble the final DataFrame from matched index lists.

    Handles column-name conflicts with *suffixes* and pads unmatched rows with
    ``NaN`` for left/right/outer joins.
    """
    suffix_l, suffix_r = suffixes
    overlap = set(left.columns) & set(right.columns)

    left_out_cols: List[str] = [
        f"{c}{suffix_l}" if c in overlap else c for c in left.columns
    ]
    right_out_cols: List[str] = [
        f"{c}{suffix_r}" if c in overlap else c for c in right.columns
    ]
    all_cols: List[str] = left_out_cols + right_out_cols

    def _select_left(idxs: List[int]) -> pd.DataFrame:
        df = left.iloc[idxs].copy()
        df.columns = pd.Index(left_out_cols)
        return df.reset_index(drop=True)

    def _select_right(idxs: List[int]) -> pd.DataFrame:
        df = right.iloc[idxs].copy()
        df.columns = pd.Index(right_out_cols)
        return df.reset_index(drop=True)

    # ── inner join part ─────────────────────────────────────────────────────
    if left_positions:
        inner: pd.DataFrame = pd.concat(
            [_select_left(left_positions), _select_right(right_positions)],
            axis=1,
        )
    else:
        inner = pd.DataFrame(columns=all_cols)

    if how == "inner":
        return inner

    parts: List[pd.DataFrame] = [inner]

    # ── unmatched left rows (left / outer join) ──────────────────────────────
    if how in ("left", "outer"):
        matched_l = set(left_positions)
        unmatched_l = [i for i in range(len(left)) if i not in matched_l]
        if unmatched_l:
            ul = _select_left(unmatched_l)
            for col in right_out_cols:
                ul[col] = np.nan
            parts.append(ul)

    # ── unmatched right rows (right / outer join) ────────────────────────────
    if how in ("right", "outer"):
        matched_r = set(right_positions)
        unmatched_r = [i for i in range(len(right)) if i not in matched_r]
        if unmatched_r:
            ur = _select_right(unmatched_r)
            for col in left_out_cols:
                ur[col] = np.nan
            parts.append(ur)

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def interval_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    lower_bound: RowPredicate,
    upper_bound: RowPredicate,
    how: JoinHow = "inner",
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    """Perform an interval join on two **pre-sorted** DataFrames.

    Both DataFrames **must** already be sorted so that the match window for
    each left row is a contiguous segment of right rows, and that segment moves
    monotonically forward as left rows advance.

    Time complexity: **O(n + m)** where *n* = ``len(left)`` and
    *m* = ``len(right)``.

    Parameters
    ----------
    left:
        Left DataFrame, pre-sorted.
    right:
        Right DataFrame, pre-sorted.
    lower_bound:
        ``lower_bound(left_row, right_row)`` returns ``True`` when *right_row*
        is at or above the lower boundary of *left_row*'s match window.
        ``False`` causes the right pointer to advance past *right_row*.
    upper_bound:
        ``upper_bound(left_row, right_row)`` returns ``True`` when *right_row*
        is within or at the upper boundary of *left_row*'s match window.
        ``False`` stops the inner scan.
    how:
        Type of join – ``'inner'`` (default), ``'left'``, ``'right'``, or
        ``'outer'``.
    suffixes:
        Suffixes appended to overlapping column names.  Defaults to
        ``('_x', '_y')``.

    Returns
    -------
    pd.DataFrame
        Joined DataFrame.

    Examples
    --------
    Join events to every point that falls inside the event's time window:

    >>> import pandas as pd
    >>> from pandas_interval_join import interval_join
    >>> events = pd.DataFrame({"start": [1, 3], "end": [5, 7], "name": ["A", "B"]})
    >>> points = pd.DataFrame({"time": [2, 4, 6], "value": [10, 20, 30]})
    >>> interval_join(
    ...     events, points,
    ...     lower_bound=lambda l, r: l["start"] <= r["time"],
    ...     upper_bound=lambda l, r: r["time"] <= l["end"],
    ... )
       start  end name  time  value
    0      1    5    A     2     10
    1      1    5    A     4     20
    2      3    7    B     4     20
    3      3    7    B     6     30
    """
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    left_pos, right_pos = _two_pointer_match(left, right, lower_bound, upper_bound)
    return _build_joined(left, right, left_pos, right_pos, how, suffixes)


def sorted_interval_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    lower_bound: RowPredicate,
    upper_bound: RowPredicate,
    left_on: List[str],
    right_on: List[str],
    how: JoinHow = "inner",
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    """Sort DataFrames by equijoin keys, then apply an interval join.

    This is a convenience wrapper around :func:`interval_join` that:

    1. Sorts both DataFrames by their respective join keys.
    2. Wraps the user-supplied predicates with a key-equality guard so that
       rows from different key groups are never matched.
    3. Removes the duplicate right-side key columns from the result.

    The equijoin predicates are handled internally, so *lower_bound* and
    *upper_bound* should only encode the **interval** part of the condition
    (the part that creates the sliding window within each key group).

    Time complexity: **O(n log n + m log m)** for sorting, then **O(n + m)**
    for the join.

    Parameters
    ----------
    left:
        Left DataFrame (need not be pre-sorted).
    right:
        Right DataFrame (need not be pre-sorted).
    lower_bound:
        Interval predicate applied when join keys match; same semantics as in
        :func:`interval_join`.
    upper_bound:
        Interval predicate applied when join keys match; same semantics as in
        :func:`interval_join`.
    left_on:
        Column names in *left* to equijoin on.
    right_on:
        Column names in *right* to equijoin on.  Must have the same length as
        *left_on*.
    how:
        Type of join.  Default ``'inner'``.
    suffixes:
        Suffixes for overlapping non-key column names.  Default ``('_x', '_y')``.

    Returns
    -------
    pd.DataFrame
        Joined DataFrame.  Each join key appears once (from the left side).

    Examples
    --------
    Join orders to trades that have the same ticker and fall in the order's
    time window:

    >>> import pandas as pd
    >>> from pandas_interval_join import sorted_interval_join
    >>> orders = pd.DataFrame({
    ...     "ticker": ["AAPL", "AAPL", "GOOG"],
    ...     "t_start": [1, 4, 1],
    ...     "t_end":   [5, 8, 3],
    ... })
    >>> trades = pd.DataFrame({
    ...     "ticker": ["AAPL", "AAPL", "AAPL", "GOOG"],
    ...     "time":   [2, 5, 7, 2],
    ...     "price":  [100, 101, 102, 50],
    ... })
    >>> sorted_interval_join(
    ...     orders, trades,
    ...     lower_bound=lambda l, r: l["t_start"] <= r["time"],
    ...     upper_bound=lambda l, r: r["time"] <= l["t_end"],
    ...     left_on=["ticker"],
    ...     right_on=["ticker"],
    ... )
      ticker  t_start  t_end  time  price
    0   AAPL        1      5     2    100
    1   AAPL        1      5     5    101
    2   AAPL        4      8     5    101
    3   AAPL        4      8     7    102
    4   GOOG        1      3     2     50
    """
    if len(left_on) != len(right_on):
        raise ValueError("`left_on` and `right_on` must have the same length.")

    left_s = left.sort_values(by=left_on, kind="stable").reset_index(drop=True)
    right_s = right.sort_values(by=right_on, kind="stable").reset_index(drop=True)

    def _key_lower(l_row: pd.Series, r_row: pd.Series) -> bool:
        """Stop advancing j when the right key group is ahead of the left."""
        for lk, rk in zip(left_on, right_on):
            lv = l_row[lk]
            rv = r_row[rk]
            if lv > rv:
                # Right row belongs to a *previous* key group → advance j.
                return False
            if lv < rv:
                # Right row belongs to a *later* key group → stop advancing j
                # (the inner scan will immediately fail the upper_bound check).
                return True
        # Keys are equal → apply the interval lower bound.
        return lower_bound(l_row, r_row)

    def _key_upper(l_row: pd.Series, r_row: pd.Series) -> bool:
        """Break the inner scan as soon as the key group changes or the
        interval upper bound is exceeded."""
        for lk, rk in zip(left_on, right_on):
            if l_row[lk] != r_row[rk]:
                return False
        # Keys are equal → apply the interval upper bound.
        return upper_bound(l_row, r_row)

    result = interval_join(left_s, right_s, _key_lower, _key_upper, how, suffixes)

    # Drop duplicate right-side key columns and restore unsuffixed left-key names.
    overlap = set(left.columns) & set(right.columns)
    cols_to_drop: List[str] = []
    for rk in right_on:
        col = f"{rk}{suffixes[1]}" if rk in overlap else rk
        if col in result.columns:
            cols_to_drop.append(col)
    if cols_to_drop:
        result = result.drop(columns=cols_to_drop)

    # If a left key column was suffixed due to overlap, rename it back.
    cols_to_rename = {
        f"{lk}{suffixes[0]}": lk
        for lk in left_on
        if f"{lk}{suffixes[0]}" in result.columns
    }
    if cols_to_rename:
        result = result.rename(columns=cols_to_rename)

    return result
