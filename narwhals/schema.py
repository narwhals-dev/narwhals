"""Schema.

Adapted from Polars implementation at:
https://github.com/pola-rs/polars/blob/main/py-polars/polars/schema.py.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import TYPE_CHECKING, cast

from narwhals._utils import Implementation, Version, zip_strict

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any, ClassVar

    import polars as pl
    import pyarrow as pa
    import sqlframe.base.types as sqlframe_types

    from narwhals._spark_like.utils import SparkSession
    from narwhals._typing import _SparkLikeImpl
    from narwhals.dtypes import DType
    from narwhals.typing import DTypeBackend


__all__ = ["Schema"]


class Schema(OrderedDict[str, "DType"]):
    """Ordered mapping of column names to their data type.

    Arguments:
        schema: The schema definition given by column names and their associated
            *instantiated* Narwhals data type. Accepts a mapping or an iterable of tuples.

    Examples:
        Define a schema by passing *instantiated* data types.

        >>> import narwhals as nw
        >>> schema = nw.Schema({"foo": nw.Int8(), "bar": nw.String()})
        >>> schema
        Schema({'foo': Int8, 'bar': String})

        Access the data type associated with a specific column name.

        >>> schema["foo"]
        Int8

        Access various schema properties using the `names`, `dtypes`, and `len` methods.

        >>> schema.names()
        ['foo', 'bar']
        >>> schema.dtypes()
        [Int8, String]
        >>> schema.len()
        2
    """

    _version: ClassVar[Version] = Version.MAIN

    def __init__(
        self, schema: Mapping[str, DType] | Iterable[tuple[str, DType]] | None = None
    ) -> None:
        schema = schema or {}
        super().__init__(schema)

    def names(self) -> list[str]:
        """Get the column names of the schema."""
        return list(self.keys())

    def dtypes(self) -> list[DType]:
        """Get the data types of the schema."""
        return list(self.values())

    def len(self) -> int:
        """Get the number of columns in the schema."""
        return len(self)

    def to_arrow(self) -> pa.Schema:
        """Convert Schema to a pyarrow Schema.

        Examples:
            >>> import narwhals as nw
            >>> schema = nw.Schema({"a": nw.Int64(), "b": nw.Datetime("ns")})
            >>> schema.to_arrow()
            a: int64
            b: timestamp[ns]
        """
        import pyarrow as pa  # ignore-banned-import

        from narwhals._arrow.utils import narwhals_to_native_dtype

        return pa.schema(
            (name, narwhals_to_native_dtype(dtype, self._version))
            for name, dtype in self.items()
        )

    def to_pandas(
        self, dtype_backend: DTypeBackend | Iterable[DTypeBackend] = None
    ) -> dict[str, Any]:
        """Convert Schema to an ordered mapping of column names to their pandas data type.

        Arguments:
            dtype_backend: Backend(s) used for the native types. When providing more than
                one, the length of the iterable must be equal to the length of the schema.

        Examples:
            >>> import narwhals as nw
            >>> schema = nw.Schema({"a": nw.Int64(), "b": nw.Datetime("ns")})
            >>> schema.to_pandas()
            {'a': 'int64', 'b': 'datetime64[ns]'}

            >>> schema.to_pandas("pyarrow")
            {'a': 'Int64[pyarrow]', 'b': 'timestamp[ns][pyarrow]'}
        """
        from narwhals._pandas_like.utils import narwhals_to_native_dtype

        to_native_dtype = partial(
            narwhals_to_native_dtype,
            implementation=Implementation.PANDAS,
            version=self._version,
        )
        if dtype_backend is None or isinstance(dtype_backend, str):
            return {
                name: to_native_dtype(dtype=dtype, dtype_backend=dtype_backend)
                for name, dtype in self.items()
            }
        backends = tuple(dtype_backend)
        if len(backends) != len(self):
            from itertools import chain, islice, repeat

            n_user, n_actual = len(backends), len(self)
            suggestion = tuple(
                islice(chain.from_iterable(islice(repeat(backends), n_actual)), n_actual)
            )
            msg = (
                f"Provided {n_user!r} `dtype_backend`(s), but schema contains {n_actual!r} field(s).\n"
                "Hint: instead of\n"
                f"    schema.to_pandas({backends})\n"
                "you may want to use\n"
                f"    schema.to_pandas({backends[0]})\n"
                f"or\n"
                f"    schema.to_pandas({suggestion})"
            )
            raise ValueError(msg)
        return {
            name: to_native_dtype(dtype=dtype, dtype_backend=backend)
            for name, dtype, backend in zip_strict(self.keys(), self.values(), backends)
        }

    def to_polars(self) -> pl.Schema:
        """Convert Schema to a polars Schema.

        Examples:
            >>> import narwhals as nw
            >>> schema = nw.Schema({"a": nw.Int64(), "b": nw.Datetime("ns")})
            >>> schema.to_polars()
            Schema({'a': Int64, 'b': Datetime(time_unit='ns', time_zone=None)})
        """
        import polars as pl  # ignore-banned-import

        from narwhals._polars.utils import narwhals_to_native_dtype

        pl_version = Implementation.POLARS._backend_version()
        schema = (
            (name, narwhals_to_native_dtype(dtype, self._version))
            for name, dtype in self.items()
        )
        return (
            pl.Schema(schema)
            if pl_version >= (1, 0, 0)
            else cast("pl.Schema", dict(schema))
        )

    def to_pyspark(self, *, session: SparkSession) -> sqlframe_types.StructType:
        return self._to_spark_like(backend=Implementation.PYSPARK, session=session)

    def to_pyspark_connect(self, *, session: SparkSession) -> sqlframe_types.StructType:
        return self._to_spark_like(
            backend=Implementation.PYSPARK_CONNECT, session=session
        )

    def to_sqlframe(self, *, session: SparkSession) -> sqlframe_types.StructType:
        return self._to_spark_like(backend=Implementation.SQLFRAME, session=session)

    def _to_spark_like(
        self, *, backend: _SparkLikeImpl, session: SparkSession
    ) -> sqlframe_types.StructType:
        from narwhals._spark_like.utils import (
            import_native_dtypes,
            narwhals_to_native_dtype as narwhals_to_spark_like_dtype,
        )

        version = self._version
        spark_dtypes = import_native_dtypes(backend)
        StructType = spark_dtypes.StructType  # noqa: N806
        StructField = spark_dtypes.StructField  # noqa: N806

        _narwhals_to_spark_like_dtype = partial(
            narwhals_to_spark_like_dtype,
            version=version,
            spark_types=spark_dtypes,
            session=session,
        )
        return StructType(  # type: ignore[no-any-return]
            [
                StructField(name, _narwhals_to_spark_like_dtype(nw_dtype), True)
                for name, nw_dtype in self.items()
            ]
        )
