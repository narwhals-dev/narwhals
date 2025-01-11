from __future__ import annotations

EXAMPLES = {
    "date": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [datetime(2012, 1, 7, 10, 20), datetime(2023, 3, 10, 11, 32)]}
            >>> df_pd = pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow")
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a library agnostic function:

            >>> def agnostic_dt_date(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(nw.col("a").dt.date()).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_date`:

            >>> agnostic_dt_date(df_pd)
                        a
            0  2012-01-07
            1  2023-03-10

            >>> agnostic_dt_date(df_pl)
            shape: (2, 1)
            ┌────────────┐
            │ a          │
            │ ---        │
            │ date       │
            ╞════════════╡
            │ 2012-01-07 │
            │ 2023-03-10 │
            └────────────┘

            >>> agnostic_dt_date(df_pa)
            pyarrow.Table
            a: date32[day]
            ----
            a: [[2012-01-07,2023-03-10]]
        """,
    "year": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 6, 1),
            ...         datetime(2024, 12, 13),
            ...         datetime(2065, 1, 1),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_year(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.year().alias("year")
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_year`:

            >>> agnostic_dt_year(df_pd)
                datetime  year
            0 1978-06-01  1978
            1 2024-12-13  2024
            2 2065-01-01  2065

            >>> agnostic_dt_year(df_pl)
            shape: (3, 2)
            ┌─────────────────────┬──────┐
            │ datetime            ┆ year │
            │ ---                 ┆ ---  │
            │ datetime[μs]        ┆ i32  │
            ╞═════════════════════╪══════╡
            │ 1978-06-01 00:00:00 ┆ 1978 │
            │ 2024-12-13 00:00:00 ┆ 2024 │
            │ 2065-01-01 00:00:00 ┆ 2065 │
            └─────────────────────┴──────┘

            >>> agnostic_dt_year(df_pa)
            pyarrow.Table
            datetime: timestamp[us]
            year: int64
            ----
            datetime: [[1978-06-01 00:00:00.000000,2024-12-13 00:00:00.000000,2065-01-01 00:00:00.000000]]
            year: [[1978,2024,2065]]
        """,
    "month": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 6, 1),
            ...         datetime(2024, 12, 13),
            ...         datetime(2065, 1, 1),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_month(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.month().alias("month"),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_month`:

            >>> agnostic_dt_month(df_pd)
                datetime  month
            0 1978-06-01      6
            1 2024-12-13     12
            2 2065-01-01      1

            >>> agnostic_dt_month(df_pl)
            shape: (3, 2)
            ┌─────────────────────┬───────┐
            │ datetime            ┆ month │
            │ ---                 ┆ ---   │
            │ datetime[μs]        ┆ i8    │
            ╞═════════════════════╪═══════╡
            │ 1978-06-01 00:00:00 ┆ 6     │
            │ 2024-12-13 00:00:00 ┆ 12    │
            │ 2065-01-01 00:00:00 ┆ 1     │
            └─────────────────────┴───────┘

            >>> agnostic_dt_month(df_pa)
            pyarrow.Table
            datetime: timestamp[us]
            month: int64
            ----
            datetime: [[1978-06-01 00:00:00.000000,2024-12-13 00:00:00.000000,2065-01-01 00:00:00.000000]]
            month: [[6,12,1]]
        """,
    "day": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 6, 1),
            ...         datetime(2024, 12, 13),
            ...         datetime(2065, 1, 1),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_day(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.day().alias("day"),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_day`:

            >>> agnostic_dt_day(df_pd)
                datetime  day
            0 1978-06-01    1
            1 2024-12-13   13
            2 2065-01-01    1

            >>> agnostic_dt_day(df_pl)
            shape: (3, 2)
            ┌─────────────────────┬─────┐
            │ datetime            ┆ day │
            │ ---                 ┆ --- │
            │ datetime[μs]        ┆ i8  │
            ╞═════════════════════╪═════╡
            │ 1978-06-01 00:00:00 ┆ 1   │
            │ 2024-12-13 00:00:00 ┆ 13  │
            │ 2065-01-01 00:00:00 ┆ 1   │
            └─────────────────────┴─────┘

            >>> agnostic_dt_day(df_pa)
            pyarrow.Table
            datetime: timestamp[us]
            day: int64
            ----
            datetime: [[1978-06-01 00:00:00.000000,2024-12-13 00:00:00.000000,2065-01-01 00:00:00.000000]]
            day: [[1,13,1]]
        """,
    "hour": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 1, 1, 1),
            ...         datetime(2024, 10, 13, 5),
            ...         datetime(2065, 1, 1, 10),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_hour(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.hour().alias("hour")
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_hour`:

            >>> agnostic_dt_hour(df_pd)
                         datetime  hour
            0 1978-01-01 01:00:00     1
            1 2024-10-13 05:00:00     5
            2 2065-01-01 10:00:00    10

            >>> agnostic_dt_hour(df_pl)
            shape: (3, 2)
            ┌─────────────────────┬──────┐
            │ datetime            ┆ hour │
            │ ---                 ┆ ---  │
            │ datetime[μs]        ┆ i8   │
            ╞═════════════════════╪══════╡
            │ 1978-01-01 01:00:00 ┆ 1    │
            │ 2024-10-13 05:00:00 ┆ 5    │
            │ 2065-01-01 10:00:00 ┆ 10   │
            └─────────────────────┴──────┘

            >>> agnostic_dt_hour(df_pa)
            pyarrow.Table
            datetime: timestamp[us]
            hour: int64
            ----
            datetime: [[1978-01-01 01:00:00.000000,2024-10-13 05:00:00.000000,2065-01-01 10:00:00.000000]]
            hour: [[1,5,10]]
        """,
    "minute": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 1, 1, 1, 1),
            ...         datetime(2024, 10, 13, 5, 30),
            ...         datetime(2065, 1, 1, 10, 20),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_minute(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.minute().alias("minute"),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_minute`:

            >>> agnostic_dt_minute(df_pd)
                         datetime  minute
            0 1978-01-01 01:01:00       1
            1 2024-10-13 05:30:00      30
            2 2065-01-01 10:20:00      20

            >>> agnostic_dt_minute(df_pl)
            shape: (3, 2)
            ┌─────────────────────┬────────┐
            │ datetime            ┆ minute │
            │ ---                 ┆ ---    │
            │ datetime[μs]        ┆ i8     │
            ╞═════════════════════╪════════╡
            │ 1978-01-01 01:01:00 ┆ 1      │
            │ 2024-10-13 05:30:00 ┆ 30     │
            │ 2065-01-01 10:20:00 ┆ 20     │
            └─────────────────────┴────────┘

            >>> agnostic_dt_minute(df_pa)
            pyarrow.Table
            datetime: timestamp[us]
            minute: int64
            ----
            datetime: [[1978-01-01 01:01:00.000000,2024-10-13 05:30:00.000000,2065-01-01 10:20:00.000000]]
            minute: [[1,30,20]]
        """,
    "second": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 1, 1, 1, 1, 1),
            ...         datetime(2024, 10, 13, 5, 30, 14),
            ...         datetime(2065, 1, 1, 10, 20, 30),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_second(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.second().alias("second"),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_second`:

            >>> agnostic_dt_second(df_pd)
                         datetime  second
            0 1978-01-01 01:01:01       1
            1 2024-10-13 05:30:14      14
            2 2065-01-01 10:20:30      30

            >>> agnostic_dt_second(df_pl)
            shape: (3, 2)
            ┌─────────────────────┬────────┐
            │ datetime            ┆ second │
            │ ---                 ┆ ---    │
            │ datetime[μs]        ┆ i8     │
            ╞═════════════════════╪════════╡
            │ 1978-01-01 01:01:01 ┆ 1      │
            │ 2024-10-13 05:30:14 ┆ 14     │
            │ 2065-01-01 10:20:30 ┆ 30     │
            └─────────────────────┴────────┘

            >>> agnostic_dt_second(df_pa)
            pyarrow.Table
            datetime: timestamp[us]
            second: int64
            ----
            datetime: [[1978-01-01 01:01:01.000000,2024-10-13 05:30:14.000000,2065-01-01 10:20:30.000000]]
            second: [[1,14,30]]
        """,
    "millisecond": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 1, 1, 1, 1, 1, 0),
            ...         datetime(2024, 10, 13, 5, 30, 14, 505000),
            ...         datetime(2065, 1, 1, 10, 20, 30, 67000),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_millisecond(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.millisecond().alias("millisecond"),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_millisecond`:

            >>> agnostic_dt_millisecond(df_pd)
                             datetime  millisecond
            0 1978-01-01 01:01:01.000            0
            1 2024-10-13 05:30:14.505          505
            2 2065-01-01 10:20:30.067           67

            >>> agnostic_dt_millisecond(df_pl)
            shape: (3, 2)
            ┌─────────────────────────┬─────────────┐
            │ datetime                ┆ millisecond │
            │ ---                     ┆ ---         │
            │ datetime[μs]            ┆ i32         │
            ╞═════════════════════════╪═════════════╡
            │ 1978-01-01 01:01:01     ┆ 0           │
            │ 2024-10-13 05:30:14.505 ┆ 505         │
            │ 2065-01-01 10:20:30.067 ┆ 67          │
            └─────────────────────────┴─────────────┘

            >>> agnostic_dt_millisecond(df_pa)
            pyarrow.Table
            datetime: timestamp[us]
            millisecond: int64
            ----
            datetime: [[1978-01-01 01:01:01.000000,2024-10-13 05:30:14.505000,2065-01-01 10:20:30.067000]]
            millisecond: [[0,505,67]]
        """,
    "microsecond": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 1, 1, 1, 1, 1, 0),
            ...         datetime(2024, 10, 13, 5, 30, 14, 505000),
            ...         datetime(2065, 1, 1, 10, 20, 30, 67000),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_microsecond(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.microsecond().alias("microsecond"),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_microsecond`:

            >>> agnostic_dt_microsecond(df_pd)
                             datetime  microsecond
            0 1978-01-01 01:01:01.000            0
            1 2024-10-13 05:30:14.505       505000
            2 2065-01-01 10:20:30.067        67000

            >>> agnostic_dt_microsecond(df_pl)
            shape: (3, 2)
            ┌─────────────────────────┬─────────────┐
            │ datetime                ┆ microsecond │
            │ ---                     ┆ ---         │
            │ datetime[μs]            ┆ i32         │
            ╞═════════════════════════╪═════════════╡
            │ 1978-01-01 01:01:01     ┆ 0           │
            │ 2024-10-13 05:30:14.505 ┆ 505000      │
            │ 2065-01-01 10:20:30.067 ┆ 67000       │
            └─────────────────────────┴─────────────┘

            >>> agnostic_dt_microsecond(df_pa)
            pyarrow.Table
            datetime: timestamp[us]
            microsecond: int64
            ----
            datetime: [[1978-01-01 01:01:01.000000,2024-10-13 05:30:14.505000,2065-01-01 10:20:30.067000]]
            microsecond: [[0,505000,67000]]
        """,
    "nanosecond": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "datetime": [
            ...         datetime(1978, 1, 1, 1, 1, 1, 0),
            ...         datetime(2024, 10, 13, 5, 30, 14, 500000),
            ...         datetime(2065, 1, 1, 10, 20, 30, 60000),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_nanosecond(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("datetime").dt.nanosecond().alias("nanosecond"),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_nanosecond`:

            >>> agnostic_dt_nanosecond(df_pd)
                             datetime  nanosecond
            0 1978-01-01 01:01:01.000           0
            1 2024-10-13 05:30:14.500   500000000
            2 2065-01-01 10:20:30.060    60000000

            >>> agnostic_dt_nanosecond(df_pl)
            shape: (3, 2)
            ┌─────────────────────────┬────────────┐
            │ datetime                ┆ nanosecond │
            │ ---                     ┆ ---        │
            │ datetime[μs]            ┆ i32        │
            ╞═════════════════════════╪════════════╡
            │ 1978-01-01 01:01:01     ┆ 0          │
            │ 2024-10-13 05:30:14.500 ┆ 500000000  │
            │ 2065-01-01 10:20:30.060 ┆ 60000000   │
            └─────────────────────────┴────────────┘

            >>> agnostic_dt_nanosecond(df_pa)
            pyarrow.Table
            datetime: timestamp[us]
            nanosecond: int64
            ----
            datetime: [[1978-01-01 01:01:01.000000,2024-10-13 05:30:14.500000,2065-01-01 10:20:30.060000]]
            nanosecond: [[0,500000000,60000000]]
        """,
    "ordinal_day": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [datetime(2020, 1, 1), datetime(2020, 8, 3)]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_ordinal_day(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         a_ordinal_day=nw.col("a").dt.ordinal_day()
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_ordinal_day`:

            >>> agnostic_dt_ordinal_day(df_pd)
                       a  a_ordinal_day
            0 2020-01-01              1
            1 2020-08-03            216

            >>> agnostic_dt_ordinal_day(df_pl)
            shape: (2, 2)
            ┌─────────────────────┬───────────────┐
            │ a                   ┆ a_ordinal_day │
            │ ---                 ┆ ---           │
            │ datetime[μs]        ┆ i16           │
            ╞═════════════════════╪═══════════════╡
            │ 2020-01-01 00:00:00 ┆ 1             │
            │ 2020-08-03 00:00:00 ┆ 216           │
            └─────────────────────┴───────────────┘

            >>> agnostic_dt_ordinal_day(df_pa)
            pyarrow.Table
            a: timestamp[us]
            a_ordinal_day: int64
            ----
            a: [[2020-01-01 00:00:00.000000,2020-08-03 00:00:00.000000]]
            a_ordinal_day: [[1,216]]
        """,
    "weekday": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [datetime(2020, 1, 1), datetime(2020, 8, 3)]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_weekday(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(a_weekday=nw.col("a").dt.weekday()).to_native()

            We can then pass either pandas, Polars, PyArrow, and other supported libraries to
            `agnostic_dt_weekday`:

            >>> agnostic_dt_weekday(df_pd)
                       a  a_weekday
            0 2020-01-01          3
            1 2020-08-03          1

            >>> agnostic_dt_weekday(df_pl)
            shape: (2, 2)
            ┌─────────────────────┬───────────┐
            │ a                   ┆ a_weekday │
            │ ---                 ┆ ---       │
            │ datetime[μs]        ┆ i8        │
            ╞═════════════════════╪═══════════╡
            │ 2020-01-01 00:00:00 ┆ 3         │
            │ 2020-08-03 00:00:00 ┆ 1         │
            └─────────────────────┴───────────┘

            >>> agnostic_dt_weekday(df_pa)
            pyarrow.Table
            a: timestamp[us]
            a_weekday: int64
            ----
            a: [[2020-01-01 00:00:00.000000,2020-08-03 00:00:00.000000]]
            a_weekday: [[3,1]]
        """,
    "total_minutes": """
            >>> from datetime import timedelta
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [timedelta(minutes=10), timedelta(minutes=20, seconds=40)]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_total_minutes(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         a_total_minutes=nw.col("a").dt.total_minutes()
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_total_minutes`:

            >>> agnostic_dt_total_minutes(df_pd)
                            a  a_total_minutes
            0 0 days 00:10:00               10
            1 0 days 00:20:40               20

            >>> agnostic_dt_total_minutes(df_pl)
            shape: (2, 2)
            ┌──────────────┬─────────────────┐
            │ a            ┆ a_total_minutes │
            │ ---          ┆ ---             │
            │ duration[μs] ┆ i64             │
            ╞══════════════╪═════════════════╡
            │ 10m          ┆ 10              │
            │ 20m 40s      ┆ 20              │
            └──────────────┴─────────────────┘

            >>> agnostic_dt_total_minutes(df_pa)
            pyarrow.Table
            a: duration[us]
            a_total_minutes: int64
            ----
            a: [[600000000,1240000000]]
            a_total_minutes: [[10,20]]
        """,
    "total_seconds": """
            >>> from datetime import timedelta
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [timedelta(seconds=10), timedelta(seconds=20, milliseconds=40)]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_total_seconds(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         a_total_seconds=nw.col("a").dt.total_seconds()
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_total_seconds`:

            >>> agnostic_dt_total_seconds(df_pd)
                                   a  a_total_seconds
            0        0 days 00:00:10               10
            1 0 days 00:00:20.040000               20

            >>> agnostic_dt_total_seconds(df_pl)
            shape: (2, 2)
            ┌──────────────┬─────────────────┐
            │ a            ┆ a_total_seconds │
            │ ---          ┆ ---             │
            │ duration[μs] ┆ i64             │
            ╞══════════════╪═════════════════╡
            │ 10s          ┆ 10              │
            │ 20s 40ms     ┆ 20              │
            └──────────────┴─────────────────┘

            >>> agnostic_dt_total_seconds(df_pa)
            pyarrow.Table
            a: duration[us]
            a_total_seconds: int64
            ----
            a: [[10000000,20040000]]
            a_total_seconds: [[10,20]]
        """,
    "total_milliseconds": """
            >>> from datetime import timedelta
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "a": [
            ...         timedelta(milliseconds=10),
            ...         timedelta(milliseconds=20, microseconds=40),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_total_milliseconds(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         a_total_milliseconds=nw.col("a").dt.total_milliseconds()
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_total_milliseconds`:

            >>> agnostic_dt_total_milliseconds(df_pd)
                                   a  a_total_milliseconds
            0 0 days 00:00:00.010000                    10
            1 0 days 00:00:00.020040                    20

            >>> agnostic_dt_total_milliseconds(df_pl)
            shape: (2, 2)
            ┌──────────────┬──────────────────────┐
            │ a            ┆ a_total_milliseconds │
            │ ---          ┆ ---                  │
            │ duration[μs] ┆ i64                  │
            ╞══════════════╪══════════════════════╡
            │ 10ms         ┆ 10                   │
            │ 20040µs      ┆ 20                   │
            └──────────────┴──────────────────────┘

            >>> agnostic_dt_total_milliseconds(df_pa)
            pyarrow.Table
            a: duration[us]
            a_total_milliseconds: int64
            ----
            a: [[10000,20040]]
            a_total_milliseconds: [[10,20]]
        """,
    "total_microseconds": """
            >>> from datetime import timedelta
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "a": [
            ...         timedelta(microseconds=10),
            ...         timedelta(milliseconds=1, microseconds=200),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_total_microseconds(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         a_total_microseconds=nw.col("a").dt.total_microseconds()
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_total_microseconds`:

            >>> agnostic_dt_total_microseconds(df_pd)
                                   a  a_total_microseconds
            0 0 days 00:00:00.000010                    10
            1 0 days 00:00:00.001200                  1200

            >>> agnostic_dt_total_microseconds(df_pl)
            shape: (2, 2)
            ┌──────────────┬──────────────────────┐
            │ a            ┆ a_total_microseconds │
            │ ---          ┆ ---                  │
            │ duration[μs] ┆ i64                  │
            ╞══════════════╪══════════════════════╡
            │ 10µs         ┆ 10                   │
            │ 1200µs       ┆ 1200                 │
            └──────────────┴──────────────────────┘

            >>> agnostic_dt_total_microseconds(df_pa)
            pyarrow.Table
            a: duration[us]
            a_total_microseconds: int64
            ----
            a: [[10,1200]]
            a_total_microseconds: [[10,1200]]
        """,
    "total_nanoseconds": """
            >>> from datetime import timedelta
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = ["2024-01-01 00:00:00.000000001", "2024-01-01 00:00:00.000000002"]
            >>> df_pd = pd.DataFrame({"a": pd.to_datetime(data)})
            >>> df_pl = pl.DataFrame({"a": data}).with_columns(
            ...     pl.col("a").str.to_datetime(time_unit="ns")
            ... )

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_total_nanoseconds(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         a_diff_total_nanoseconds=nw.col("a").diff().dt.total_nanoseconds()
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_total_nanoseconds`:

            >>> agnostic_dt_total_nanoseconds(df_pd)
                                          a  a_diff_total_nanoseconds
            0 2024-01-01 00:00:00.000000001                       NaN
            1 2024-01-01 00:00:00.000000002                       1.0

            >>> agnostic_dt_total_nanoseconds(df_pl)
            shape: (2, 2)
            ┌───────────────────────────────┬──────────────────────────┐
            │ a                             ┆ a_diff_total_nanoseconds │
            │ ---                           ┆ ---                      │
            │ datetime[ns]                  ┆ i64                      │
            ╞═══════════════════════════════╪══════════════════════════╡
            │ 2024-01-01 00:00:00.000000001 ┆ null                     │
            │ 2024-01-01 00:00:00.000000002 ┆ 1                        │
            └───────────────────────────────┴──────────────────────────┘
        """,
    "to_string": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "a": [
            ...         datetime(2020, 3, 1),
            ...         datetime(2020, 4, 1),
            ...         datetime(2020, 5, 1),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_dt_to_string(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("a").dt.to_string("%Y/%m/%d %H:%M:%S")
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_to_string`:

            >>> agnostic_dt_to_string(df_pd)
                                 a
            0  2020/03/01 00:00:00
            1  2020/04/01 00:00:00
            2  2020/05/01 00:00:00

            >>> agnostic_dt_to_string(df_pl)
            shape: (3, 1)
            ┌─────────────────────┐
            │ a                   │
            │ ---                 │
            │ str                 │
            ╞═════════════════════╡
            │ 2020/03/01 00:00:00 │
            │ 2020/04/01 00:00:00 │
            │ 2020/05/01 00:00:00 │
            └─────────────────────┘

            >>> agnostic_dt_to_string(df_pa)
            pyarrow.Table
            a: string
            ----
            a: [["2020/03/01 00:00:00.000000","2020/04/01 00:00:00.000000","2020/05/01 00:00:00.000000"]]
        """,
    "replace_time_zone": """
            >>> from datetime import datetime, timezone
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "a": [
            ...         datetime(2024, 1, 1, tzinfo=timezone.utc),
            ...         datetime(2024, 1, 2, tzinfo=timezone.utc),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_dt_replace_time_zone(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("a").dt.replace_time_zone("Asia/Kathmandu")
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_replace_time_zone`:

            >>> agnostic_dt_replace_time_zone(df_pd)
                                      a
            0 2024-01-01 00:00:00+05:45
            1 2024-01-02 00:00:00+05:45

            >>> agnostic_dt_replace_time_zone(df_pl)
            shape: (2, 1)
            ┌──────────────────────────────┐
            │ a                            │
            │ ---                          │
            │ datetime[μs, Asia/Kathmandu] │
            ╞══════════════════════════════╡
            │ 2024-01-01 00:00:00 +0545    │
            │ 2024-01-02 00:00:00 +0545    │
            └──────────────────────────────┘

            >>> agnostic_dt_replace_time_zone(df_pa)
            pyarrow.Table
            a: timestamp[us, tz=Asia/Kathmandu]
            ----
            a: [[2023-12-31 18:15:00.000000Z,2024-01-01 18:15:00.000000Z]]
        """,
    "convert_time_zone": """
            >>> from datetime import datetime, timezone
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {
            ...     "a": [
            ...         datetime(2024, 1, 1, tzinfo=timezone.utc),
            ...         datetime(2024, 1, 2, tzinfo=timezone.utc),
            ...     ]
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_dt_convert_time_zone(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("a").dt.convert_time_zone("Asia/Kathmandu")
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_convert_time_zone`:

            >>> agnostic_dt_convert_time_zone(df_pd)
                                      a
            0 2024-01-01 05:45:00+05:45
            1 2024-01-02 05:45:00+05:45

            >>> agnostic_dt_convert_time_zone(df_pl)
            shape: (2, 1)
            ┌──────────────────────────────┐
            │ a                            │
            │ ---                          │
            │ datetime[μs, Asia/Kathmandu] │
            ╞══════════════════════════════╡
            │ 2024-01-01 05:45:00 +0545    │
            │ 2024-01-02 05:45:00 +0545    │
            └──────────────────────────────┘

            >>> agnostic_dt_convert_time_zone(df_pa)
            pyarrow.Table
            a: timestamp[us, tz=Asia/Kathmandu]
            ----
            a: [[2024-01-01 00:00:00.000000Z,2024-01-02 00:00:00.000000Z]]
        """,
    "timestamp": """
            >>> from datetime import date
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"date": [date(2001, 1, 1), None, date(2001, 1, 3)]}
            >>> df_pd = pd.DataFrame(data, dtype="datetime64[ns]")
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            Let's define a dataframe-agnostic function:

            >>> def agnostic_dt_timestamp(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         nw.col("date").dt.timestamp().alias("timestamp_us"),
            ...         nw.col("date").dt.timestamp("ms").alias("timestamp_ms"),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_dt_timestamp`:

            >>> agnostic_dt_timestamp(df_pd)
                    date  timestamp_us  timestamp_ms
            0 2001-01-01  9.783072e+14  9.783072e+11
            1        NaT           NaN           NaN
            2 2001-01-03  9.784800e+14  9.784800e+11

            >>> agnostic_dt_timestamp(df_pl)
            shape: (3, 3)
            ┌────────────┬─────────────────┬──────────────┐
            │ date       ┆ timestamp_us    ┆ timestamp_ms │
            │ ---        ┆ ---             ┆ ---          │
            │ date       ┆ i64             ┆ i64          │
            ╞════════════╪═════════════════╪══════════════╡
            │ 2001-01-01 ┆ 978307200000000 ┆ 978307200000 │
            │ null       ┆ null            ┆ null         │
            │ 2001-01-03 ┆ 978480000000000 ┆ 978480000000 │
            └────────────┴─────────────────┴──────────────┘

            >>> agnostic_dt_timestamp(df_pa)
            pyarrow.Table
            date: date32[day]
            timestamp_us: int64
            timestamp_ms: int64
            ----
            date: [[2001-01-01,null,2001-01-03]]
            timestamp_us: [[978307200000000,null,978480000000000]]
            timestamp_ms: [[978307200000,null,978480000000]]
        """,
}
