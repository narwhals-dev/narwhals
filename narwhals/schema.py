"""Schema.

Adapted from Polars implementation at:
https://github.com/pola-rs/polars/blob/main/py-polars/polars/schema.py.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, cast

from narwhals._utils import Implementation, Version, qualified_type_name
from narwhals.dependencies import (
    get_cudf,
    is_cudf_dtype,
    is_pandas_like_dtype,
    is_polars_data_type,
    is_polars_schema,
    is_pyarrow_data_type,
    is_pyarrow_schema,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, ClassVar

    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.typing import (
        DTypeBackend,
        IntoArrowSchema,
        IntoPandasSchema,
        IntoPolarsSchema,
    )


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
        """Get the column names of the schema.

        Returns:
            Column names.
        """
        return list(self.keys())

    def dtypes(self) -> list[DType]:
        """Get the data types of the schema.

        Returns:
            Data types of schema.
        """
        return list(self.values())

    def len(self) -> int:
        """Get the number of columns in the schema.

        Returns:
            Number of columns.
        """
        return len(self)

    @classmethod
    def from_arrow(cls, schema: IntoArrowSchema, /) -> Self:
        """Construct a Schema from a pyarrow Schema.

        Arguments:
            schema: A pyarrow Schema or non-empty mapping of column names to
                pyarrow data types.

        Returns:
            A Narwhals Schema.
        """
        import pyarrow as pa  # ignore-banned-import

        from narwhals._arrow.utils import native_to_narwhals_dtype

        if isinstance(schema, Mapping):
            schema = pa.schema(schema)
        return cls(
            (field.name, native_to_narwhals_dtype(field.type, cls._version))
            for field in schema
        )

    @classmethod
    def from_cudf(cls, schema: IntoPandasSchema, /) -> Self:  # pragma: no cover
        """Construct a Schema from a cudf schema representation.

        Arguments:
            schema: A non-empty mapping of column names to cudf data types.

        Returns:
            A Narwhals Schema.
        """
        return cls._from_pandas_like(schema, Implementation.CUDF)

    @classmethod
    def from_pandas(cls, schema: IntoPandasSchema, /) -> Self:
        """Construct a Schema from a pandas schema representation.

        Arguments:
            schema: A non-empty mapping of column names to pandas data types.

        Returns:
            A Narwhals Schema.
        """
        return cls._from_pandas_like(schema, Implementation.PANDAS)

    @classmethod
    def from_pandas_like(cls, schema: IntoPandasSchema, /) -> Self:
        """Construct a Schema from a pandas-like schema representation.

        Arguments:
            schema: A non-empty mapping of column names to pandas-like data types.

        Returns:
            A Narwhals Schema.
        """
        from_native = (
            cls.from_cudf
            if get_cudf() and any(is_cudf_dtype(dtype) for dtype in schema.values())
            else cls.from_pandas
        )
        return from_native(schema)

    # NOTE: Just aliasing for consistency
    # Only `cuDF` has unique paths
    from_modin = from_pandas

    @classmethod
    def from_native(
        cls, schema: IntoArrowSchema | IntoPolarsSchema | IntoPandasSchema, /
    ) -> Self:
        """Construct a Schema from a native schema representation.

        Arguments:
            schema: A native schema object, or non-empty mapping of column names to
                *instantiated* native data types.

        Returns:
            A Narwhals Schema.
        """
        if is_pyarrow_schema(schema):
            return cls.from_arrow(schema)
        if is_polars_schema(schema):
            return cls.from_polars(schema)
        # avoid the empty case as well, since we have no dtypes to sample
        if isinstance(schema, Mapping) and schema:
            return cls._from_native_mapping(schema)
        msg = (
            f"Expected an arrow, polars, or pandas schema, but got "
            f"{qualified_type_name(schema)!r}\n\n{schema!r}"
        )
        raise TypeError(msg)

    @classmethod
    def from_polars(cls, schema: IntoPolarsSchema, /) -> Self:
        """Construct a Schema from a polars Schema.

        Arguments:
            schema: A polars Schema or non-empty mapping of column names to
                *instantiated* polars data types.

        Returns:
            A Narwhals Schema.
        """
        from narwhals._polars.utils import native_to_narwhals_dtype

        return cls(
            (name, native_to_narwhals_dtype(dtype, cls._version))
            for name, dtype in schema.items()
        )

    def to_arrow(self) -> pa.Schema:
        """Convert Schema to a pyarrow Schema.

        Returns:
            A pyarrow Schema.

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

        Returns:
            An ordered mapping of column names to their pandas data type.

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
            for name, dtype, backend in zip(self.keys(), self.values(), backends)
        }

    def to_polars(self) -> pl.Schema:
        """Convert Schema to a polars Schema.

        Returns:
            A polars Schema or plain dict (prior to polars 1.0).

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

    @classmethod
    def _from_native_mapping(
        cls,
        native: Mapping[str, pa.DataType] | Mapping[str, pl.DataType] | IntoPandasSchema,
        /,
    ) -> Self:
        first_item = next(iter(native.items()))
        first_key, first_dtype = first_item
        if is_polars_data_type(first_dtype):
            native_polars = cast("IntoPolarsSchema", native)
            return cls.from_polars(native_polars)
        if is_pandas_like_dtype(first_dtype):
            native_pandas = cast("IntoPandasSchema", native)
            return cls.from_pandas_like(native_pandas)
        if is_pyarrow_data_type(first_dtype):
            native_arrow = cast("IntoArrowSchema", native)
            return cls.from_arrow(native_arrow)
        msg = (
            f"Expected an arrow, polars, or pandas dtype, but found "
            f"`{first_key}: {qualified_type_name(first_dtype)}`\n\n{native!r}"
        )
        raise TypeError(msg)

    @classmethod
    def _from_pandas_like(
        cls, schema: IntoPandasSchema, implementation: Implementation, /
    ) -> Self:
        from narwhals._pandas_like.utils import (
            native_to_narwhals_dtype,
            object_native_to_narwhals_dtype,
        )

        impl = implementation
        is_object = impl.to_native_namespace().api.types.is_object_dtype
        from_native = partial(
            native_to_narwhals_dtype, implementation=impl, version=cls._version
        )
        object_dtype = partial(
            object_native_to_narwhals_dtype, (), implementation=impl, version=cls._version
        )
        return cls(
            (name, (object_dtype() if is_object(dtype) else from_native(dtype)))
            for name, dtype in schema.items()
        )
