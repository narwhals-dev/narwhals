from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence

from sqlalchemy import Column

from narwhals._expression_parsing import infer_new_root_output_names
from narwhals._sqlalchemy.utils import maybe_evaluate
from narwhals.typing import CompliantExpr
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._sqlalchemy.dataframe import SQLAlchemyLazyFrame
    from narwhals._sqlalchemy.namespace import SQLAlchemyNamespace
    from narwhals.utils import Version


class SQLAlchemyExpr(CompliantExpr[Column]):
    _implementation = Implementation.SQLALCHEMY

    def __init__(
        self: Self,
        call: Callable[[SQLAlchemyLazyFrame], Sequence[Column]],
        *,
        depth: int,
        function_name: str,
        root_names: list[str] | None,
        output_names: list[str] | None,
        # Whether the expression is a length-1 Column resulting from
        # a reduction, such as `nw.col('a').sum()`
        returns_scalar: bool,
        backend_version: tuple[int, ...],
        version: Version,
        kwargs: dict[str, Any],
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._root_names = root_names
        self._output_names = output_names
        self._returns_scalar = returns_scalar
        self._backend_version = backend_version
        self._version = version
        self._kwargs = kwargs

    def __call__(self: Self, df: SQLAlchemyLazyFrame) -> Sequence[Column]:
        return self._call(df)

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> SQLAlchemyNamespace:  # pragma: no cover
        from narwhals._sqlalchemy.namespace import SQLAlchemyNamespace

        return SQLAlchemyNamespace(
            backend_version=self._backend_version, version=self._version
        )

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(_: SQLAlchemyLazyFrame) -> list[Column]:
            return [Column(col_name) for col_name in column_names]

        return cls(
            func,
            depth=0,
            function_name="col",
            root_names=list(column_names),
            output_names=list(column_names),
            returns_scalar=False,
            backend_version=backend_version,
            version=version,
            kwargs={},
        )

    @classmethod
    def from_column_indices(
        cls: type[Self],
        *column_indices: int,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(df: SQLAlchemyLazyFrame) -> list[Column]:
            columns = df.columns

            return [Column(columns[i]) for i in column_indices]

        return cls(
            func,
            depth=0,
            function_name="nth",
            root_names=None,
            output_names=None,
            returns_scalar=False,
            backend_version=backend_version,
            version=version,
            kwargs={},
        )

    def _from_call(
        self: Self,
        call: Callable[..., Column],
        expr_name: str,
        *,
        returns_scalar: bool,
        **kwargs: Any,
    ) -> Self:
        def func(df: SQLAlchemyLazyFrame) -> list[Column]:
            inputs = self._call(df)
            _kwargs = {key: maybe_evaluate(df, value) for key, value in kwargs.items()}
            return [call(_input, **_kwargs) for _input in inputs]

        root_names, output_names = infer_new_root_output_names(self, **kwargs)

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=f"{self._function_name}->{expr_name}",
            root_names=root_names,
            output_names=output_names,
            returns_scalar=returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
            kwargs=kwargs,
        )
