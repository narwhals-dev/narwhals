from __future__ import annotations

EXAMPLES = {
    "date": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> dates = [datetime(2012, 1, 7, 10, 20), datetime(2023, 3, 10, 11, 32)]
            >>> s_pd = pd.Series(dates).convert_dtypes(dtype_backend="pyarrow")
            >>> s_pl = pl.Series(dates)
            >>> s_pa = pa.chunked_array([dates])

            We define a library agnostic function:

            >>> def agnostic_date(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.date().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_date`:

            >>> agnostic_date(s_pd)
            0    2012-01-07
            1    2023-03-10
            dtype: date32[day][pyarrow]

            >>> agnostic_date(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [date]
            [
               2012-01-07
               2023-03-10
            ]

            >>> agnostic_date(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2012-01-07,
                2023-03-10
              ]
            ]
        """,
    "year": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> dates = [datetime(2012, 1, 7), datetime(2023, 3, 10)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)
            >>> s_pa = pa.chunked_array([dates])

            We define a library agnostic function:

            >>> def agnostic_year(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.year().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_year`:

            >>> agnostic_year(s_pd)
            0    2012
            1    2023
            dtype: int...

            >>> agnostic_year(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i32]
            [
               2012
               2023
            ]

            >>> agnostic_year(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2012,
                2023
              ]
            ]
        """,
    "month": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> dates = [datetime(2023, 2, 1), datetime(2023, 8, 3)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)
            >>> s_pa = pa.chunked_array([dates])

            We define a library agnostic function:

            >>> def agnostic_month(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.month().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_month`:

            >>> agnostic_month(s_pd)
            0    2
            1    8
            dtype: int...
            >>> agnostic_month(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i8]
            [
               2
               8
            ]

            >>> agnostic_month(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2,
                8
              ]
            ]
        """,
    "day": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> dates = [datetime(2022, 1, 1), datetime(2022, 1, 5)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)
            >>> s_pa = pa.chunked_array([dates])

            We define a library agnostic function:

            >>> def agnostic_day(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.day().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_day`:

            >>> agnostic_day(s_pd)
            0    1
            1    5
            dtype: int...

            >>> agnostic_day(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i8]
            [
               1
               5
            ]

            >>> agnostic_day(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                5
              ]
            ]
        """,
    "hour": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> dates = [datetime(2022, 1, 1, 5, 3), datetime(2022, 1, 5, 9, 12)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)
            >>> s_pa = pa.chunked_array([dates])

            We define a library agnostic function:

            >>> def agnostic_hour(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.hour().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_hour`:

            >>> agnostic_hour(s_pd)
            0    5
            1    9
            dtype: int...

            >>> agnostic_hour(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i8]
            [
               5
               9
            ]

            >>> agnostic_hour(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                5,
                9
              ]
            ]
        """,
    "minute": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> dates = [datetime(2022, 1, 1, 5, 3), datetime(2022, 1, 5, 9, 12)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)
            >>> s_pa = pa.chunked_array([dates])

            We define a library agnostic function:

            >>> def agnostic_minute(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.minute().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_minute`:

            >>> agnostic_minute(s_pd)
            0     3
            1    12
            dtype: int...

            >>> agnostic_minute(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i8]
            [
               3
               12
            ]

            >>> agnostic_minute(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                3,
                12
              ]
            ]
        """,
    "second": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> dates = [datetime(2022, 1, 1, 5, 3, 10), datetime(2022, 1, 5, 9, 12, 4)]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)
            >>> s_pa = pa.chunked_array([dates])

            We define a library agnostic function:

            >>> def agnostic_second(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.second().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_second`:

            >>> agnostic_second(s_pd)
            0    10
            1     4
            dtype: int...

            >>> agnostic_second(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i8]
            [
               10
                4
            ]

            >>> agnostic_second(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                10,
                4
              ]
            ]
        """,
    "millisecond": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> dates = [
            ...     datetime(2023, 5, 21, 12, 55, 10, 400000),
            ...     datetime(2023, 5, 21, 12, 55, 10, 600000),
            ...     datetime(2023, 5, 21, 12, 55, 10, 800000),
            ...     datetime(2023, 5, 21, 12, 55, 11, 0),
            ...     datetime(2023, 5, 21, 12, 55, 11, 200000),
            ... ]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)
            >>> s_pa = pa.chunked_array([dates])

            We define a library agnostic function:

            >>> def agnostic_millisecond(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.millisecond().alias("datetime").to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_millisecond`:

            >>> agnostic_millisecond(s_pd)
            0    400
            1    600
            2    800
            3      0
            4    200
            Name: datetime, dtype: int...

            >>> agnostic_millisecond(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: 'datetime' [i32]
            [
                400
                600
                800
                0
                200
            ]

            >>> agnostic_millisecond(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                400,
                600,
                800,
                0,
                200
              ]
            ]
        """,
    "microsecond": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> dates = [
            ...     datetime(2023, 5, 21, 12, 55, 10, 400000),
            ...     datetime(2023, 5, 21, 12, 55, 10, 600000),
            ...     datetime(2023, 5, 21, 12, 55, 10, 800000),
            ...     datetime(2023, 5, 21, 12, 55, 11, 0),
            ...     datetime(2023, 5, 21, 12, 55, 11, 200000),
            ... ]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)
            >>> s_pa = pa.chunked_array([dates])

            We define a library agnostic function:

            >>> def agnostic_microsecond(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.microsecond().alias("datetime").to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_microsecond`:

            >>> agnostic_microsecond(s_pd)
            0    400000
            1    600000
            2    800000
            3         0
            4    200000
            Name: datetime, dtype: int...

            >>> agnostic_microsecond(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (5,)
            Series: 'datetime' [i32]
            [
               400000
               600000
               800000
               0
               200000
            ]

            >>> agnostic_microsecond(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                400000,
                600000,
                800000,
                0,
                200000
              ]
            ]
        """,
    "nanosecond": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> dates = [
            ...     datetime(2022, 1, 1, 5, 3, 10, 500000),
            ...     datetime(2022, 1, 5, 9, 12, 4, 60000),
            ... ]
            >>> s_pd = pd.Series(dates)
            >>> s_pl = pl.Series(dates)
            >>> s_pa = pa.chunked_array([dates])

            We define a library agnostic function:

            >>> def agnostic_nanosecond(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.nanosecond().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_nanosecond`:

            >>> agnostic_nanosecond(s_pd)
            0    500000000
            1     60000000
            dtype: int...

            >>> agnostic_nanosecond(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i32]
            [
               500000000
               60000000
            ]

            >>> agnostic_nanosecond(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                500000000,
                60000000
              ]
            ]
        """,
    "ordinal_day": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [datetime(2020, 1, 1), datetime(2020, 8, 3)]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_ordinal_day(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.ordinal_day().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_ordinal_day`:

            >>> agnostic_ordinal_day(s_pd)
            0      1
            1    216
            dtype: int32

            >>> agnostic_ordinal_day(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i16]
            [
               1
               216
            ]


            >>> agnostic_ordinal_day(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                1,
                216
              ]
            ]
        """,
    "weekday": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT
            >>> data = [datetime(2020, 1, 1), datetime(2020, 8, 3)]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_weekday(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.weekday().to_native()

            We can then pass either pandas, Polars, PyArrow, and other supported libraries to `agnostic_weekday`:

            >>> agnostic_weekday(s_pd)
            0    3
            1    1
            dtype: int32
            >>> agnostic_weekday(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i8]
            [
               3
               1
            ]
            >>> agnostic_weekday(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                3,
                1
              ]
            ]
        """,
    "total_minutes": """
            >>> from datetime import timedelta
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [timedelta(minutes=10), timedelta(minutes=20, seconds=40)]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_total_minutes(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.total_minutes().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_total_minutes`:

            >>> agnostic_total_minutes(s_pd)
            0    10
            1    20
            dtype: int...

            >>> agnostic_total_minutes(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i64]
            [
                    10
                    20
            ]

            >>> agnostic_total_minutes(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                10,
                20
              ]
            ]
        """,
    "total_seconds": """
            >>> from datetime import timedelta
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [timedelta(seconds=10), timedelta(seconds=20, milliseconds=40)]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_total_seconds(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.total_seconds().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_total_seconds`:

            >>> agnostic_total_seconds(s_pd)
            0    10
            1    20
            dtype: int...

            >>> agnostic_total_seconds(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i64]
            [
                    10
                    20
            ]

            >>> agnostic_total_seconds(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                10,
                20
              ]
            ]
        """,
    "total_milliseconds": """
            >>> from datetime import timedelta
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [
            ...     timedelta(milliseconds=10),
            ...     timedelta(milliseconds=20, microseconds=40),
            ... ]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_total_milliseconds(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.total_milliseconds().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_total_milliseconds`:

            >>> agnostic_total_milliseconds(s_pd)
            0    10
            1    20
            dtype: int...

            >>> agnostic_total_milliseconds(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i64]
            [
                    10
                    20
            ]

            >>> agnostic_total_milliseconds(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                10,
                20
              ]
            ]
        """,
    "total_microseconds": """
            >>> from datetime import timedelta
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [
            ...     timedelta(microseconds=10),
            ...     timedelta(milliseconds=1, microseconds=200),
            ... ]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a library agnostic function:

            >>> def agnostic_total_microseconds(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.total_microseconds().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_total_microseconds`:

            >>> agnostic_total_microseconds(s_pd)
            0      10
            1    1200
            dtype: int...

            >>> agnostic_total_microseconds(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i64]
            [
                10
                1200
            ]

            >>> agnostic_total_microseconds(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                10,
                1200
              ]
            ]
        """,
    "total_nanoseconds": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = ["2024-01-01 00:00:00.000000001", "2024-01-01 00:00:00.000000002"]
            >>> s_pd = pd.to_datetime(pd.Series(data))
            >>> s_pl = pl.Series(data).str.to_datetime(time_unit="ns")

            We define a library agnostic function:

            >>> def agnostic_total_nanoseconds(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.diff().dt.total_nanoseconds().to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_total_nanoseconds`:

            >>> agnostic_total_nanoseconds(s_pd)
            0    NaN
            1    1.0
            dtype: float64

            >>> agnostic_total_nanoseconds(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [i64]
            [
                    null
                    1
            ]
        """,
    "to_string": """
            >>> from datetime import datetime
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [
            ...     datetime(2020, 3, 1),
            ...     datetime(2020, 4, 1),
            ...     datetime(2020, 5, 1),
            ... ]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            We define a dataframe-agnostic function:

            >>> def agnostic_to_string(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.to_string("%Y/%m/%d").to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_to_string`:

            >>> agnostic_to_string(s_pd)
            0    2020/03/01
            1    2020/04/01
            2    2020/05/01
            dtype: object

            >>> agnostic_to_string(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [str]
            [
               "2020/03/01"
               "2020/04/01"
               "2020/05/01"
            ]

            >>> agnostic_to_string(s_pa)  # doctest: +ELLIPSIS
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                "2020/03/01",
                "2020/04/01",
                "2020/05/01"
              ]
            ]
        """,
    "replace_time_zone": """
            >>> from datetime import datetime, timezone
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [
            ...     datetime(2024, 1, 1, tzinfo=timezone.utc),
            ...     datetime(2024, 1, 2, tzinfo=timezone.utc),
            ... ]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_replace_time_zone(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.replace_time_zone("Asia/Kathmandu").to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_replace_time_zone`:

            >>> agnostic_replace_time_zone(s_pd)
            0   2024-01-01 00:00:00+05:45
            1   2024-01-02 00:00:00+05:45
            dtype: datetime64[ns, Asia/Kathmandu]

            >>> agnostic_replace_time_zone(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [datetime[μs, Asia/Kathmandu]]
            [
                2024-01-01 00:00:00 +0545
                2024-01-02 00:00:00 +0545
            ]

            >>> agnostic_replace_time_zone(s_pa)
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2023-12-31 18:15:00.000000Z,
                2024-01-01 18:15:00.000000Z
              ]
            ]
        """,
    "convert_time_zone": """
            >>> from datetime import datetime, timezone
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [
            ...     datetime(2024, 1, 1, tzinfo=timezone.utc),
            ...     datetime(2024, 1, 2, tzinfo=timezone.utc),
            ... ]
            >>> s_pd = pd.Series(data)
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_convert_time_zone(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.convert_time_zone("Asia/Kathmandu").to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_convert_time_zone`:

            >>> agnostic_convert_time_zone(s_pd)
            0   2024-01-01 05:45:00+05:45
            1   2024-01-02 05:45:00+05:45
            dtype: datetime64[ns, Asia/Kathmandu]

            >>> agnostic_convert_time_zone(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (2,)
            Series: '' [datetime[μs, Asia/Kathmandu]]
            [
                2024-01-01 05:45:00 +0545
                2024-01-02 05:45:00 +0545
            ]

            >>> agnostic_convert_time_zone(s_pa)
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                2024-01-01 00:00:00.000000Z,
                2024-01-02 00:00:00.000000Z
              ]
            ]
        """,
    "timestamp": """
            >>> from datetime import date
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoSeriesT

            >>> data = [date(2001, 1, 1), None, date(2001, 1, 3)]
            >>> s_pd = pd.Series(data, dtype="datetime64[ns]")
            >>> s_pl = pl.Series(data)
            >>> s_pa = pa.chunked_array([data])

            Let's define a dataframe-agnostic function:

            >>> def agnostic_timestamp(s_native: IntoSeriesT) -> IntoSeriesT:
            ...     s = nw.from_native(s_native, series_only=True)
            ...     return s.dt.timestamp("ms").to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_timestamp`:

            >>> agnostic_timestamp(s_pd)
            0    9.783072e+11
            1             NaN
            2    9.784800e+11
            dtype: float64

            >>> agnostic_timestamp(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3,)
            Series: '' [i64]
            [
                    978307200000
                    null
                    978480000000
            ]

            >>> agnostic_timestamp(s_pa)
            <pyarrow.lib.ChunkedArray object at ...>
            [
              [
                978307200000,
                null,
                978480000000
              ]
            ]
        """,
}
