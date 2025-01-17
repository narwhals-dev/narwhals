from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals.expr import Expr
from narwhals.utils import flatten

if TYPE_CHECKING:
    from collections.abc import Collection
    from datetime import timezone

    from narwhals.typing import TimeUnit


class Selector(Expr): ...


def by_dtype(*dtypes: Any) -> Expr:
    """Select columns based on their dtype.

    Arguments:
        dtypes: one or data types to select

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [4.1, 2.3]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function to select int64 and float64
        dtypes and multiplies each value by 2:

        >>> def agnostic_select_by_dtype(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = nw.from_native(df_native)
        ...     return df_nw.select(ncs.by_dtype(nw.Int64, nw.Float64) * 2).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_select_by_dtype`:

        >>> agnostic_select_by_dtype(df_pd)
           a    c
        0  2  8.2
        1  4  4.6

        >>> agnostic_select_by_dtype(df_pl)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ c   │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 2   ┆ 8.2 │
        │ 4   ┆ 4.6 │
        └─────┴─────┘

        >>> agnostic_select_by_dtype(df_pa)
        pyarrow.Table
        a: int64
        c: double
        ----
        a: [[2,4]]
        c: [[8.2,4.6]]
    """
    return Selector(lambda plx: plx.selectors.by_dtype(flatten(dtypes)))


def numeric() -> Expr:
    """Select numeric columns.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [4.1, 2.3]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function to select numeric
        dtypes and multiplies each value by 2:

        >>> def agnostic_select_numeric(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = nw.from_native(df_native)
        ...     return df_nw.select(ncs.numeric() * 2).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_select_numeric`:

        >>> agnostic_select_numeric(df_pd)
           a    c
        0  2  8.2
        1  4  4.6

        >>> agnostic_select_numeric(df_pl)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ c   │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 2   ┆ 8.2 │
        │ 4   ┆ 4.6 │
        └─────┴─────┘

        >>> agnostic_select_numeric(df_pa)
        pyarrow.Table
        a: int64
        c: double
        ----
        a: [[2,4]]
        c: [[8.2,4.6]]
    """
    return Selector(lambda plx: plx.selectors.numeric())


def boolean() -> Expr:
    """Select boolean columns.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [False, True]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function to select boolean dtypes:

        >>> def agnostic_select_boolean(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = nw.from_native(df_native)
        ...     return df_nw.select(ncs.boolean()).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_select_boolean`:

        >>> agnostic_select_boolean(df_pd)
               c
        0  False
        1   True

        >>> agnostic_select_boolean(df_pl)
        shape: (2, 1)
        ┌───────┐
        │ c     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ false │
        │ true  │
        └───────┘

        >>> agnostic_select_boolean(df_pa)
        pyarrow.Table
        c: bool
        ----
        c: [[false,true]]
    """
    return Selector(lambda plx: plx.selectors.boolean())


def string() -> Expr:
    """Select string columns.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [False, True]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function to select string dtypes:

        >>> def agnostic_select_string(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = nw.from_native(df_native)
        ...     return df_nw.select(ncs.string()).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_select_string`:

        >>> agnostic_select_string(df_pd)
           b
        0  x
        1  y

        >>> agnostic_select_string(df_pl)
        shape: (2, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ str │
        ╞═════╡
        │ x   │
        │ y   │
        └─────┘

        >>> agnostic_select_string(df_pa)
        pyarrow.Table
        b: string
        ----
        b: [["x","y"]]
    """
    return Selector(lambda plx: plx.selectors.string())


def categorical() -> Expr:
    """Select categorical columns.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [False, True]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function that first converts column "b" to
        categorical, and then selects categorical dtypes:

        >>> def agnostic_select_categorical(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = nw.from_native(df_native).with_columns(
        ...         b=nw.col("b").cast(nw.Categorical())
        ...     )
        ...     return df_nw.select(ncs.categorical()).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_select_categorical`:

        >>> agnostic_select_categorical(df_pd)
           b
        0  x
        1  y

        >>> agnostic_select_categorical(df_pl)
        shape: (2, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ cat │
        ╞═════╡
        │ x   │
        │ y   │
        └─────┘

        >>> agnostic_select_categorical(df_pa)
        pyarrow.Table
        b: dictionary<values=string, indices=uint32, ordered=0>
        ----
        b: [  -- dictionary:
        ["x","y"]  -- indices:
        [0,1]]
    """
    return Selector(lambda plx: plx.selectors.categorical())


def all() -> Expr:
    """Select all columns.

    Returns:
        A new expression.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> data = {"a": [1, 2], "b": ["x", "y"], "c": [False, True]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function to select all dtypes:

        >>> def agnostic_select_all(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = nw.from_native(df_native)
        ...     return df_nw.select(ncs.all()).to_native()

        We can then pass any supported library such as pandas, Polars, or
        PyArrow to `agnostic_select_all`:

        >>> agnostic_select_all(df_pd)
           a  b      c
        0  1  x  False
        1  2  y   True

        >>> agnostic_select_all(df_pl)
        shape: (2, 3)
        ┌─────┬─────┬───────┐
        │ a   ┆ b   ┆ c     │
        │ --- ┆ --- ┆ ---   │
        │ i64 ┆ str ┆ bool  │
        ╞═════╪═════╪═══════╡
        │ 1   ┆ x   ┆ false │
        │ 2   ┆ y   ┆ true  │
        └─────┴─────┴───────┘

        >>> agnostic_select_all(df_pa)
        pyarrow.Table
        a: int64
        b: string
        c: bool
        ----
        a: [[1,2]]
        b: [["x","y"]]
        c: [[false,true]]
    """
    return Selector(lambda plx: plx.selectors.all())


def datetime(
    time_unit: TimeUnit | Collection[TimeUnit] | None = None,
    time_zone: str | timezone | Collection[str | timezone | None] | None = ("*", None),
) -> Expr:
    """Select all datetime columns, optionally filtering by time unit/zone.

    Arguments:
        time_unit: One (or more) of the allowed timeunit precision strings, "ms", "us",
            "ns" and "s". Omit to select columns with any valid timeunit.
        time_zone: Specify which timezone(s) to select:

            * One or more timezone strings, as defined in zoneinfo (to see valid options
                run `import zoneinfo; zoneinfo.available_timezones()` for a full list).
            * Set `None` to select Datetime columns that do not have a timezone.
            * Set `"*"` to select Datetime columns that have *any* timezone.

    Returns:
        A new expression.

    Examples:
        >>> from __future__ import annotations
        >>> from datetime import datetime, timezone
        >>> from zoneinfo import ZoneInfo
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import narwhals.selectors as ncs
        >>> from narwhals.typing import IntoFrameT
        >>>
        >>> berlin_tz = ZoneInfo("Europe/Berlin")
        >>> utc_tz = timezone.utc
        >>> data = {
        ...     "tstamp_berlin": [
        ...         datetime(1999, 7, 21, 5, 20, 16, 987654, tzinfo=berlin_tz),
        ...         datetime(2000, 5, 16, 6, 21, 21, 123465, tzinfo=berlin_tz),
        ...     ],
        ...     "tstamp_utc": [
        ...         datetime(2023, 4, 10, 12, 14, 16, 999000, tzinfo=utc_tz),
        ...         datetime(2025, 8, 25, 14, 18, 22, 666000, tzinfo=utc_tz),
        ...     ],
        ...     "tstamp": [
        ...         datetime(2000, 11, 20, 18, 12, 16, 600000),
        ...         datetime(2020, 10, 30, 10, 20, 25, 123000),
        ...     ],
        ...     "numeric": [3.14, 6.28],
        ... }
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.DataFrame(data)
        >>> df_pa = pa.table(data)

        Let's define a dataframe-agnostic function to select datetime dtypes:

        >>> def agnostic_datetime_selector(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = (
        ...         nw.from_native(df_native)
        ...         .with_columns(
        ...             tstamp_berlin=nw.col("tstamp_berlin").cast(
        ...                 nw.Datetime(time_zone="Europe/Berlin")
        ...             )
        ...         )
        ...         .select(ncs.datetime())
        ...     )
        ...     return df_nw.to_native()

        Select all datetime columns:

        >>> pd.set_option("display.width", 0)
        >>> agnostic_datetime_selector(df_pd)
                             tstamp_berlin                       tstamp_utc                  tstamp
        0 1999-07-21 05:20:16.987654+02:00 2023-04-10 12:14:16.999000+00:00 2000-11-20 18:12:16.600
        1 2000-05-16 06:21:21.123465+02:00 2025-08-25 14:18:22.666000+00:00 2020-10-30 10:20:25.123

        >>> agnostic_datetime_selector(df_pl)
        shape: (2, 3)
        ┌─────────────────────────────────┬─────────────────────────────┬─────────────────────────┐
        │ tstamp_berlin                   ┆ tstamp_utc                  ┆ tstamp                  │
        │ ---                             ┆ ---                         ┆ ---                     │
        │ datetime[μs, Europe/Berlin]     ┆ datetime[μs, UTC]           ┆ datetime[μs]            │
        ╞═════════════════════════════════╪═════════════════════════════╪═════════════════════════╡
        │ 1999-07-21 05:20:16.987654 CES… ┆ 2023-04-10 12:14:16.999 UTC ┆ 2000-11-20 18:12:16.600 │
        │ 2000-05-16 06:21:21.123465 CES… ┆ 2025-08-25 14:18:22.666 UTC ┆ 2020-10-30 10:20:25.123 │
        └─────────────────────────────────┴─────────────────────────────┴─────────────────────────┘

        >>> agnostic_datetime_selector(df_pa)
        pyarrow.Table
        tstamp_berlin: timestamp[us, tz=Europe/Berlin]
        tstamp_utc: timestamp[us, tz=UTC]
        tstamp: timestamp[us]
        ----
        tstamp_berlin: [[1999-07-21 05:20:16.987654Z,2000-05-16 06:21:21.123465Z]]
        tstamp_utc: [[2023-04-10 12:14:16.999000Z,2025-08-25 14:18:22.666000Z]]
        tstamp: [[2000-11-20 18:12:16.600000,2020-10-30 10:20:25.123000]]

        Select all datetime columns that have any time_zone specification:

        >>> def agnostic_datetime_selector_any_tz(df_native: IntoFrameT) -> IntoFrameT:
        ...     df_nw = (
        ...         nw.from_native(df_native)
        ...         .with_columns(
        ...             tstamp_berlin=nw.col("tstamp_berlin").cast(
        ...                 nw.Datetime(time_zone="Europe/Berlin")
        ...             )
        ...         )
        ...         .select(ncs.datetime(time_zone="*"))
        ...     )
        ...     return df_nw.to_native()

        >>> agnostic_datetime_selector_any_tz(df_pd)
                             tstamp_berlin                       tstamp_utc
        0 1999-07-21 05:20:16.987654+02:00 2023-04-10 12:14:16.999000+00:00
        1 2000-05-16 06:21:21.123465+02:00 2025-08-25 14:18:22.666000+00:00

        >>> agnostic_datetime_selector_any_tz(df_pl)
        shape: (2, 2)
        ┌─────────────────────────────────┬─────────────────────────────┐
        │ tstamp_berlin                   ┆ tstamp_utc                  │
        │ ---                             ┆ ---                         │
        │ datetime[μs, Europe/Berlin]     ┆ datetime[μs, UTC]           │
        ╞═════════════════════════════════╪═════════════════════════════╡
        │ 1999-07-21 05:20:16.987654 CES… ┆ 2023-04-10 12:14:16.999 UTC │
        │ 2000-05-16 06:21:21.123465 CES… ┆ 2025-08-25 14:18:22.666 UTC │
        └─────────────────────────────────┴─────────────────────────────┘

        >>> agnostic_datetime_selector_any_tz(df_pa)
        pyarrow.Table
        tstamp_berlin: timestamp[us, tz=Europe/Berlin]
        tstamp_utc: timestamp[us, tz=UTC]
        ----
        tstamp_berlin: [[1999-07-21 05:20:16.987654Z,2000-05-16 06:21:21.123465Z]]
        tstamp_utc: [[2023-04-10 12:14:16.999000Z,2025-08-25 14:18:22.666000Z]]
    """
    return Selector(
        lambda plx: plx.selectors.datetime(time_unit=time_unit, time_zone=time_zone)
    )


__all__ = [
    "all",
    "boolean",
    "by_dtype",
    "categorical",
    "numeric",
    "string",
]
