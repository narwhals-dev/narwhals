from __future__ import annotations

EXAMPLES = {
    "Decimal": """
        >>> import polars as pl
        >>> import narwhals as nw
        >>> s = pl.Series(["1.5"], dtype=pl.Decimal)
        >>> nw.from_native(s, series_only=True).dtype
        Decimal
    """,
    "Int64": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [2, 1, 3, 7]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> nw.from_native(ser_pd, series_only=True).dtype
        Int64
        >>> nw.from_native(ser_pl, series_only=True).dtype
        Int64
        >>> nw.from_native(ser_pa, series_only=True).dtype
        Int64
    """,
    "Int32": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [2, 1, 3, 7]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> def func(ser):
        ...     ser_nw = nw.from_native(ser, series_only=True)
        ...     return ser_nw.cast(nw.Int32).dtype

        >>> func(ser_pd)
        Int32
        >>> func(ser_pl)
        Int32
        >>> func(ser_pa)
        Int32
    """,
    "Int16": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [2, 1, 3, 7]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> def func(ser):
        ...     ser_nw = nw.from_native(ser, series_only=True)
        ...     return ser_nw.cast(nw.Int16).dtype

        >>> func(ser_pd)
        Int16
        >>> func(ser_pl)
        Int16
        >>> func(ser_pa)
        Int16
    """,
    "Int8": """
       >>> import pandas as pd
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> data = [2, 1, 3, 7]
       >>> ser_pd = pd.Series(data)
       >>> ser_pl = pl.Series(data)
       >>> ser_pa = pa.chunked_array([data])

       >>> def func(ser):
       ...     ser_nw = nw.from_native(ser, series_only=True)
       ...     return ser_nw.cast(nw.Int8).dtype

       >>> func(ser_pd)
       Int8
       >>> func(ser_pl)
       Int8
       >>> func(ser_pa)
       Int8
    """,
    "UInt64": """
       >>> import pandas as pd
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> data = [2, 1, 3, 7]
       >>> ser_pd = pd.Series(data)
       >>> ser_pl = pl.Series(data)
       >>> ser_pa = pa.chunked_array([data])

       >>> def func(ser):
       ...     ser_nw = nw.from_native(ser, series_only=True)
       ...     return ser_nw.cast(nw.UInt64).dtype

       >>> func(ser_pd)
       UInt64
       >>> func(ser_pl)
       UInt64
       >>> func(ser_pa)
       UInt64
    """,
    "UInt32": """
       >>> import pandas as pd
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> data = [2, 1, 3, 7]
       >>> ser_pd = pd.Series(data)
       >>> ser_pl = pl.Series(data)
       >>> ser_pa = pa.chunked_array([data])

       >>> def func(ser):
       ...     ser_nw = nw.from_native(ser, series_only=True)
       ...     return ser_nw.cast(nw.UInt32).dtype

       >>> func(ser_pd)
       UInt32
       >>> func(ser_pl)
       UInt32
       >>> func(ser_pa)
       UInt32
    """,
    "UInt16": """
       >>> import pandas as pd
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> data = [2, 1, 3, 7]
       >>> ser_pd = pd.Series(data)
       >>> ser_pl = pl.Series(data)
       >>> ser_pa = pa.chunked_array([data])

       >>> def func(ser):
       ...     ser_nw = nw.from_native(ser, series_only=True)
       ...     return ser_nw.cast(nw.UInt16).dtype

       >>> func(ser_pd)
       UInt16
       >>> func(ser_pl)
       UInt16
       >>> func(ser_pa)
       UInt16
    """,
    "UInt8": """
       >>> import pandas as pd
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> data = [2, 1, 3, 7]
       >>> ser_pd = pd.Series(data)
       >>> ser_pl = pl.Series(data)
       >>> ser_pa = pa.chunked_array([data])

       >>> def func(ser):
       ...     ser_nw = nw.from_native(ser, series_only=True)
       ...     return ser_nw.cast(nw.UInt8).dtype

       >>> func(ser_pd)
       UInt8
       >>> func(ser_pl)
       UInt8
       >>> func(ser_pa)
       UInt8
    """,
    "Float64": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [0.001, 0.1, 0.01, 0.1]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> nw.from_native(ser_pd, series_only=True).dtype
        Float64
        >>> nw.from_native(ser_pl, series_only=True).dtype
        Float64
        >>> nw.from_native(ser_pa, series_only=True).dtype
        Float64
    """,
    "Float32": """
       >>> import pandas as pd
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> data = [0.001, 0.1, 0.01, 0.1]
       >>> ser_pd = pd.Series(data)
       >>> ser_pl = pl.Series(data)
       >>> ser_pa = pa.chunked_array([data])

       >>> def func(ser):
       ...     ser_nw = nw.from_native(ser, series_only=True)
       ...     return ser_nw.cast(nw.Float32).dtype

       >>> func(ser_pd)
       Float32
       >>> func(ser_pl)
       Float32
       >>> func(ser_pa)
       Float32
    """,
    "String": """
       >>> import pandas as pd
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> data = ["beluga", "narwhal", "orca", "vaquita"]
       >>> ser_pd = pd.Series(data)
       >>> ser_pl = pl.Series(data)
       >>> ser_pa = pa.chunked_array([data])

       >>> nw.from_native(ser_pd, series_only=True).dtype
       String
       >>> nw.from_native(ser_pl, series_only=True).dtype
       String
       >>> nw.from_native(ser_pa, series_only=True).dtype
       String
    """,
    "Boolean": """
       >>> import pandas as pd
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> data = [True, False, False, True]
       >>> ser_pd = pd.Series(data)
       >>> ser_pl = pl.Series(data)
       >>> ser_pa = pa.chunked_array([data])

       >>> nw.from_native(ser_pd, series_only=True).dtype
       Boolean
       >>> nw.from_native(ser_pl, series_only=True).dtype
       Boolean
       >>> nw.from_native(ser_pa, series_only=True).dtype
       Boolean
    """,
    "Object": """
       >>> import pandas as pd
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> class Foo: ...
       >>> ser_pd = pd.Series([Foo(), Foo()])
       >>> ser_pl = pl.Series([Foo(), Foo()])

       >>> nw.from_native(ser_pd, series_only=True).dtype
       Object
       >>> nw.from_native(ser_pl, series_only=True).dtype
       Object
    """,
    "Unknown": """
       >>> import pandas as pd
       >>> import narwhals as nw
       >>> data = pd.period_range("2000-01", periods=4, freq="M")
       >>> ser_pd = pd.Series(data)

       >>> nw.from_native(ser_pd, series_only=True).dtype
       Unknown
    """,
    "Datetime": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import pyarrow.compute as pc
        >>> import narwhals as nw
        >>> from datetime import datetime, timedelta
        >>> data = [datetime(2024, 12, 9) + timedelta(days=n) for n in range(5)]
        >>> ser_pd = (
        ...     pd.Series(data)
        ...     .dt.tz_localize("Africa/Accra")
        ...     .astype("datetime64[ms, Africa/Accra]")
        ... )
        >>> ser_pl = (
        ...     pl.Series(data).cast(pl.Datetime("ms")).dt.replace_time_zone("Africa/Accra")
        ... )
        >>> ser_pa = pc.assume_timezone(
        ...     pa.chunked_array([data], type=pa.timestamp("ms")), "Africa/Accra"
        ... )

        >>> nw.from_native(ser_pd, series_only=True).dtype
        Datetime(time_unit='ms', time_zone='Africa/Accra')
        >>> nw.from_native(ser_pl, series_only=True).dtype
        Datetime(time_unit='ms', time_zone='Africa/Accra')
        >>> nw.from_native(ser_pa, series_only=True).dtype
        Datetime(time_unit='ms', time_zone='Africa/Accra')
    """,
    "Duration": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> from datetime import timedelta
        >>> data = [timedelta(seconds=d) for d in range(1, 4)]
        >>> ser_pd = pd.Series(data).astype("timedelta64[ms]")
        >>> ser_pl = pl.Series(data).cast(pl.Duration("ms"))
        >>> ser_pa = pa.chunked_array([data], type=pa.duration("ms"))

        >>> nw.from_native(ser_pd, series_only=True).dtype
        Duration(time_unit='ms')
        >>> nw.from_native(ser_pl, series_only=True).dtype
        Duration(time_unit='ms')
        >>> nw.from_native(ser_pa, series_only=True).dtype
        Duration(time_unit='ms')
    """,
    "Categorical": """
       >>> import pandas as pd
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> data = ["beluga", "narwhal", "orca", "vaquita"]
       >>> ser_pd = pd.Series(data)
       >>> ser_pl = pl.Series(data)
       >>> ser_pa = pa.chunked_array([data])

       >>> nw.from_native(ser_pd, series_only=True).cast(nw.Categorical).dtype
       Categorical
       >>> nw.from_native(ser_pl, series_only=True).cast(nw.Categorical).dtype
       Categorical
       >>> nw.from_native(ser_pa, series_only=True).cast(nw.Categorical).dtype
       Categorical
    """,
    "Enum": """
       >>> import polars as pl
       >>> import narwhals as nw
       >>> data = ["beluga", "narwhal", "orca", "vaquita"]
       >>> ser_pl = pl.Series(data, dtype=pl.Enum(data))

       >>> nw.from_native(ser_pl, series_only=True).dtype
       Enum
    """,
    "Struct": """
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> data = [{"a": 1, "b": ["narwhal", "beluga"]}, {"a": 2, "b": ["orca"]}]
       >>> ser_pl = pl.Series(data)
       >>> ser_pa = pa.chunked_array([data])

       >>> nw.from_native(ser_pl, series_only=True).dtype
       Struct({'a': Int64, 'b': List(String)})
       >>> nw.from_native(ser_pa, series_only=True).dtype
       Struct({'a': Int64, 'b': List(String)})
    """,
    "List": """
       >>> import pandas as pd
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> data = [["narwhal", "orca"], ["beluga", "vaquita"]]
       >>> ser_pd = pd.Series(data, dtype=pd.ArrowDtype(pa.large_list(pa.large_string())))
       >>> ser_pl = pl.Series(data)
       >>> ser_pa = pa.chunked_array([data])

       >>> nw.from_native(ser_pd, series_only=True).dtype
       List(String)
       >>> nw.from_native(ser_pl, series_only=True).dtype
       List(String)
       >>> nw.from_native(ser_pa, series_only=True).dtype
       List(String)
    """,
    "Array": """
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [[1, 2], [3, 4], [5, 6]]
        >>> ser_pd = pd.Series(data, dtype=pd.ArrowDtype(pa.list_(pa.int32(), 2)))
        >>> ser_pl = pl.Series(data, dtype=pl.Array(pl.Int32, 2))
        >>> ser_pa = pa.chunked_array([data], type=pa.list_(pa.int32(), 2))

        >>> nw.from_native(ser_pd, series_only=True).dtype
        Array(Int32, 2)
        >>> nw.from_native(ser_pl, series_only=True).dtype
        Array(Int32, 2)
        >>> nw.from_native(ser_pa, series_only=True).dtype
        Array(Int32, 2)
    """,
    "Date": """
       >>> import pandas as pd
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> from datetime import date, timedelta
       >>> data = [date(2024, 12, 1) + timedelta(days=d) for d in range(4)]
       >>> ser_pd = pd.Series(data, dtype="date32[pyarrow]")
       >>> ser_pl = pl.Series(data)
       >>> ser_pa = pa.chunked_array([data])

       >>> nw.from_native(ser_pd, series_only=True).dtype
       Date
       >>> nw.from_native(ser_pl, series_only=True).dtype
       Date
       >>> nw.from_native(ser_pa, series_only=True).dtype
       Date
    """,
}
