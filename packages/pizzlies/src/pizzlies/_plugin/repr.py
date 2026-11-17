"""Polars-style table reprs for `pizzlies` frames and series."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from narwhals.dtypes import DType

MAX_ROWS = 10
MAX_COLS = 8
MAX_WIDTH = 25
ELLIPSIS = "…"

_SHORT_DTYPE_NAMES = {
    "Int8": "i8",
    "Int16": "i16",
    "Int32": "i32",
    "Int64": "i64",
    "Int128": "i128",
    "UInt8": "u8",
    "UInt16": "u16",
    "UInt32": "u32",
    "UInt64": "u64",
    "UInt128": "u128",
    "Float16": "f16",
    "Float32": "f32",
    "Float64": "f64",
    "String": "str",
    "Boolean": "bool",
    "Date": "date",
    "Time": "time",
    "Binary": "binary",
    "Categorical": "cat",
    "Object": "object",
    "Unknown": "unknown",
}


def _dtype_name(dtype: Any) -> str:
    name = type(dtype).__name__
    if name in {"List", "Array"}:
        return f"{name.lower()}[{_dtype_name(dtype.inner)}]"
    if name == "Struct":
        return f"struct[{len(dtype.fields)}]"
    if name in {"Datetime", "Duration"}:
        tz = getattr(dtype, "time_zone", None)
        return f"{name.lower()}[{dtype.time_unit}{f', {tz}' if tz else ''}]"
    return _SHORT_DTYPE_NAMES.get(name, name.lower())


def _truncate(text: str) -> str:
    return text if len(text) <= MAX_WIDTH else f"{text[: MAX_WIDTH - 1]}{ELLIPSIS}"


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return _truncate(f'"{value}"')
    if isinstance(value, list):
        return _truncate(f"[{', '.join(_format_value(v) for v in value)}]")
    if isinstance(value, dict):
        fields = ", ".join(_format_value(v) for v in value.values())
        return _truncate(f"{{{fields}}}")
    return _truncate(str(value))


def _row_indices(n_rows: int) -> list[int | None]:
    """Row positions to display, with `None` standing for the elided middle."""
    if n_rows <= MAX_ROWS:
        return list(range(n_rows))
    half = MAX_ROWS // 2
    return [*range(half), None, *range(n_rows - half, n_rows)]


def _format_column(values: Sequence[Any], rows: Sequence[int | None]) -> list[str]:
    return [ELLIPSIS if i is None else _format_value(values[i]) for i in rows]


def frame_repr(native: Mapping[str, Sequence[Any]], schema: Mapping[str, DType]) -> str:
    names = list(native)
    if not names:
        return "shape: (0, 0)\n┌┐\n└┘"
    n_rows = len(native[names[0]])
    rows = _row_indices(n_rows)
    columns: list[list[str]] = [
        [
            _truncate(name),
            "---",
            _dtype_name(schema[name]),
            *_format_column(native[name], rows),
        ]
        for name in names
    ]
    if len(columns) > MAX_COLS:
        half = MAX_COLS // 2
        elided = [ELLIPSIS, "---", ELLIPSIS, *(ELLIPSIS for _ in rows)]
        columns = [*columns[:half], elided, *columns[-half:]]
    widths = [max(len(cell) for cell in column) for column in columns]

    def line(cells: Sequence[str]) -> str:
        padded = (cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
        return f"│ {' ┆ '.join(padded)} │"

    def rule(left: str, fill: str, sep: str, right: str) -> str:
        return left + sep.join(fill * (w + 2) for w in widths) + right

    header = [line([column[i] for column in columns]) for i in range(3)]
    body = [line([column[i] for column in columns]) for i in range(3, 3 + len(rows))]
    table = [
        rule("┌", "─", "┬", "┐"),
        *header,
        rule("╞", "═", "╪", "╡"),
        *body,
        rule("└", "─", "┴", "┘"),
    ]
    return "\n".join([f"shape: ({n_rows}, {len(names)})", *table])


def series_repr(name: str, values: Sequence[Any], dtype: DType) -> str:
    rows = _row_indices(len(values))
    body = (f"\t{cell}" for cell in _format_column(values, rows))
    return "\n".join(
        [
            f"shape: ({len(values)},)",
            f"Series: '{name}' [{_dtype_name(dtype)}]",
            "[",
            *body,
            "]",
        ]
    )
