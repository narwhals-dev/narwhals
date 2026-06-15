from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from narwhals._plan import expressions as ir, meta
from narwhals._plan._immutable import Immutable
from narwhals._utils import unstable

if TYPE_CHECKING:
    from narwhals._plan import Expr, Selector
    from narwhals._plan.typing import SerdeFormat, SerdeSink


class ExprMetaNamespace(Immutable):
    """Methods to traverse and introspect existing expressions."""

    __slots__ = ("_expr",)
    _expr: Expr

    # TODO @dangotbanned: docs: Rephrase the admoninition to avoid 46334x negations 😰
    def has_multiple_outputs(self) -> bool:
        """Indicate if this expression **can** expand into multiple expressions.

        Important:
            Guarantees the expression **will not** expand in the negative case,
            see ([polars#23708](https://github.com/pola-rs/polars/issues/23708))
        """
        return meta.has_selectors(self._expr._ir)

    def is_column(self) -> bool:
        """Indicate if this expression is a basic/unaliased column."""
        return isinstance(self._expr._ir, ir.Column)

    def is_column_selection(self, *, allow_aliasing: bool = False) -> bool:
        """Indicate if this expression only selects columns (optionally aliased).

        Arguments:
            allow_aliasing: If False (default), any aliasing is not considered to be column selection.
        """
        return meta.is_column_selection(self._expr._ir, allow_aliasing=allow_aliasing)

    def is_literal(self, *, allow_aliasing: bool = False) -> bool:
        """Indicate if this expression is a literal value (optionally aliased).

        Arguments:
            allow_aliasing: If False (default), only a bare literal will match.
        """
        return meta.is_literal(self._expr._ir, allow_aliasing=allow_aliasing)

    @overload
    def output_name(self, *, raise_if_undetermined: Literal[True] = True) -> str: ...
    @overload
    def output_name(self, *, raise_if_undetermined: Literal[False]) -> str | None: ...
    def output_name(self, *, raise_if_undetermined: bool = True) -> str | None:
        """Get the column name that this expression would produce.

        Arguments:
            raise_if_undetermined: If True (default), a `ComputeError` will be raised
                if the output name depends on the schema of the context.
                Otherwise `None` is returned.

        Examples:
            >>> import narwhals._plan as nw
            >>> import narwhals._plan.selectors as ncs
            >>> nw.col("a").meta.output_name()
            'a'

            Aliasing is supported:
            >>> nw.col("a").alias("b").meta.output_name()
            'b'

            And chained renaming operations:
            >>> nw.col("a").alias("b").min().name.to_uppercase().meta.output_name()
            'B'

            But selectors always require a schema:
            >>> ncs.string().meta.output_name(raise_if_undetermined=False)
            >>> ncs.string().meta.output_name()  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ComputeError: unable to find root column name for expr 'ncs.string()' when calling 'output_name'
        """
        return meta.output_name(
            self._expr._ir, raise_if_undetermined=raise_if_undetermined
        )

    def root_names(self) -> list[str]:
        """Get the root column names."""
        return list(meta.iter_root_names(self._expr._ir))

    @unstable
    def as_selector(self) -> Selector:
        """Try to turn this expression into a selector.

        Raises if the underlying expression is not a column or selector.
        """
        return self._expr._ir.to_selector_ir().to_narwhals(self._expr.version)

    # TODO @dangotbanned: Implement `ExprMetaNamespace.undo_aliases`
    # Also make add a type parameter for (exclusively) this return
    @unstable
    def undo_aliases(self) -> Expr | Selector:
        """Undo any renaming operation like `alias` or `name.keep`."""
        msg = "`meta.undo_aliases()` is not yet implemented"
        raise NotImplementedError(msg)

    # TODO @dangotbanned: Add examples
    # TODO @dangotbanned: Add unstable note
    @overload
    @unstable
    def serialize(
        self, file: None = ..., *, format: Literal["binary"] = ...
    ) -> bytes: ...
    @overload
    @unstable
    def serialize(self, file: None = ..., *, format: Literal["json"]) -> str: ...
    @overload
    @unstable
    def serialize(self, file: SerdeSink, *, format: SerdeFormat = ...) -> None: ...
    @unstable
    def serialize(
        self, file: SerdeSink | None = None, *, format: SerdeFormat = "binary"
    ) -> bytes | str | None:
        """Serialize this expression to a file or string in JSON format.

        Arguments:
            file: File path to which the result should be written.

                If set to `None` (default), the output is returned as a string instead.
            format: The format in which to serialize. Options:

                - `"binary"`: Serialize to bytes (default).
                - `"json"`: Serialize to string.
        """
        from narwhals._plan.io import serde

        if format == "json":
            return serde.serialize_json(self._expr, file)
        return serde.serialize_binary(self._expr, file)
