# pandas-interval-join

**Efficient O(n + m) interval / range join for sorted pandas DataFrames.**

[![PyPI](https://img.shields.io/pypi/v/pandas-interval-join)](https://pypi.org/project/pandas-interval-join/)
[![Python](https://img.shields.io/pypi/pyversions/pandas-interval-join)](https://pypi.org/project/pandas-interval-join/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is an interval join?

An **interval join** (also called a range join or band join) finds every pair of rows
`(left[i], right[k])` where some **ordered condition** holds between the two rows.

A classic example: match every event to all time-series points that fall inside the
event's time window:

```
left.start  ≤  right.time  ≤  left.end
```

A naïve nested-loop join is **O(n × m)**. This library exploits the fact that, when
both DataFrames are sorted appropriately, the set of valid right-side matches for each
left row forms a **contiguous, forward-moving window** — a property that allows the
join to run in **O(n + m)** time using a two-pointer sweep.

---

## Installation

```bash
pip install pandas-interval-join
```

---

## Quick start

### `interval_join` — pre-sorted DataFrames

When you already have both DataFrames sorted in the right order, use `interval_join`
directly. It runs in **O(n + m)**.

```python
import pandas as pd
from pandas_interval_join import interval_join

# Events sorted by start time
events = pd.DataFrame({
    "start": [1, 3, 6],
    "end":   [5, 7, 9],
    "label": ["A", "B", "C"],
})

# Sensor readings sorted by time
readings = pd.DataFrame({
    "time":  [2, 4, 6, 8],
    "value": [10, 20, 30, 40],
})

result = interval_join(
    events,
    readings,
    lower_bound=lambda l, r: l["start"] <= r["time"],
    upper_bound=lambda l, r: r["time"]  <= l["end"],
)
print(result)
#    start  end label  time  value
# 0      1    5     A     2     10
# 1      1    5     A     4     20
# 2      3    7     B     4     20
# 3      3    7     B     6     30
# 4      6    9     C     6     30
# 5      6    9     C     8     40
```

### `sorted_interval_join` — automatic sorting + equijoin on keys

If you additionally want to **equijoin on key columns** (e.g., same ticker symbol)
before applying the interval condition, use `sorted_interval_join`. It sorts both
DataFrames by the given keys and then applies the two-pointer sweep within each
key group. Total complexity: **O(n log n + m log m)**.

```python
import pandas as pd
from pandas_interval_join import sorted_interval_join

orders = pd.DataFrame({
    "ticker":  ["AAPL", "AAPL", "GOOG"],
    "t_start": [1,      4,      1],
    "t_end":   [5,      8,      3],
})

trades = pd.DataFrame({
    "ticker": ["AAPL", "AAPL", "AAPL", "GOOG"],
    "time":   [2,      5,      7,      2],
    "price":  [100,    101,    102,    50],
})

result = sorted_interval_join(
    orders,
    trades,
    lower_bound=lambda l, r: l["t_start"] <= r["time"],
    upper_bound=lambda l, r: r["time"]    <= l["t_end"],
    left_on=["ticker"],
    right_on=["ticker"],
)
print(result)
#   ticker  t_start  t_end  time  price
# 0   AAPL        1      5     2    100
# 1   AAPL        1      5     5    101
# 2   AAPL        4      8     5    101
# 3   AAPL        4      8     7    102
# 4   GOOG        1      3     2     50
```

---

## API reference

### `interval_join`

```python
interval_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    lower_bound: Callable[[pd.Series, pd.Series], bool],
    upper_bound: Callable[[pd.Series, pd.Series], bool],
    how: str = "inner",            # 'inner' | 'left' | 'right' | 'outer'
    suffixes: tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame
```

**Pre-conditions (must be satisfied by the caller):**

1. Both DataFrames are sorted so that the match window for each left row is a
   **contiguous** segment of right rows.
2. As left rows advance, that segment moves **monotonically forward** in right.

**Predicate semantics:**

| Predicate | Returns `True` when… | Effect when `False` |
|---|---|---|
| `lower_bound(left_row, right_row)` | `right_row` is at or above the window's lower bound | Right pointer advances past `right_row` |
| `upper_bound(left_row, right_row)` | `right_row` is within or at the window's upper bound | Inner scan stops |

---

### `sorted_interval_join`

```python
sorted_interval_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    lower_bound: Callable[[pd.Series, pd.Series], bool],
    upper_bound: Callable[[pd.Series, pd.Series], bool],
    left_on: list[str],
    right_on: list[str],
    how: str = "inner",
    suffixes: tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame
```

Sorts both DataFrames by `left_on` / `right_on`, wraps the predicates with a
key-equality check, runs `interval_join`, and removes the duplicate right-side
key columns from the result.

**Important:** Within each key group the rows must be ordered so that the interval
window property holds. `sorted_interval_join` only sorts by the equijoin keys; you
are responsible for ordering the interval columns within each group.

---

### `how` parameter (join type)

| Value | Behaviour |
|---|---|
| `"inner"` (default) | Only matched rows |
| `"left"` | All left rows; unmatched get `NaN` for right columns |
| `"right"` | All right rows; unmatched get `NaN` for left columns |
| `"outer"` | All rows from both sides |

---

## How the algorithm works

The algorithm is a **two-pointer sweep**:

```
j = 0  # rolling start of the right-side window

for each left row i:
    # advance j past right rows that are BELOW the window
    while j < len(right) and not lower_bound(left[i], right[j]):
        j += 1

    # collect all right rows INSIDE the window
    previous_j = j
    k = j
    while k < len(right) and upper_bound(left[i], right[k]):
        emit pair (i, k)
        k += 1

    # restore j; the next left row's window starts no earlier than previous_j
    j = previous_j
```

Each right row is visited at most twice (once by the outer `while`, once by the inner
`while`), giving **O(n + m)** total comparisons.

---

## Requirements

- Python ≥ 3.9
- pandas ≥ 1.3
- numpy ≥ 1.21

---

## Development

```bash
# install in editable mode with dev extras
pip install -e ".[dev]"

# run tests (includes mypy type check)
pytest

# type-check the package source
mypy pandas_interval_join --strict
```

---

## License

MIT — see [LICENSE](LICENSE).
