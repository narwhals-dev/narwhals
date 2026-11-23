"""Polars-style table reprs for `pizzlies` frames and series."""

from __future__ import annotations

import os
import unicodedata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from narwhals.dtypes import DType

MAX_ROWS = 10
MAX_COLS = 8
MAX_WIDTH = 25
CELL_WIDTHS = (None, 15, 8)
"""Widths cells are cut to (`None`: not cut), narrower ones only if wider ones don't fit."""
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


def available_width() -> int:
    """The width a repr can use inside Narwhals' box, before it gives up on showing it."""
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = int(os.getenv("COLUMNS", "80"))
    return width - 2


_ESCAPES = {
    i: "\\n" if i == 10 else "\\t" if i == 9 else "\\r" if i == 13 else f"\\x{i:02x}"
    for i in [*range(32), 127]
}
"""Control characters would break the table (and Narwhals expands tabs), so escape them."""


def _escape(text: str) -> str:
    return text.translate(_ESCAPES)


def _char_width(char: str) -> int:
    if unicodedata.combining(char) or unicodedata.category(char) in {"Mn", "Me", "Cf"}:
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def display_width(text: str) -> int:
    """The number of terminal columns `text` takes, which `len` gets wrong for e.g. CJK."""
    return len(text) if text.isascii() else sum(map(_char_width, text))


def _truncate(text: str, width: int | None = MAX_WIDTH) -> str:
    if width is None or display_width(text) <= width:
        return text
    kept, used = [], 0
    for char in text:
        used += _char_width(char)
        if used > width - 1:
            break
        kept.append(char)
    return f"{''.join(kept)}{ELLIPSIS}"


def _ljust(text: str, width: int) -> str:
    return text + " " * (width - display_width(text))


def fit_lines(text: str, width: int | None = None) -> str:
    """Cut each line of `text` to `width` (by default, `available_width()`)."""
    width = available_width() if width is None else width
    return "\n".join(_truncate(line, width) for line in text.splitlines())


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return _truncate(f'"{_escape(value)}"')
    if isinstance(value, list):
        return _truncate(f"[{', '.join(_format_value(v) for v in value)}]")
    if isinstance(value, dict):
        fields = ", ".join(_format_value(v) for v in value.values())
        return _truncate(f"{{{fields}}}")
    # Like Polars, other scalars (e.g. datetimes with microseconds) are shown in full.
    return _escape(str(value))


def _row_indices(n_rows: int) -> list[int | None]:
    """Row positions to display, with `None` standing for the elided middle."""
    if n_rows <= MAX_ROWS:
        return list(range(n_rows))
    half = MAX_ROWS // 2
    return [*range(half), None, *range(n_rows - half, n_rows)]


def _format_column(values: Sequence[Any], rows: Sequence[int | None]) -> list[str]:
    return [ELLIPSIS if i is None else _format_value(values[i]) for i in rows]


def _table_width(widths: Sequence[int]) -> int:
    """The width of a table with these column widths, borders and separators included."""
    return sum(widths) + 3 * len(widths) + 1


def _shown_columns(widths: Sequence[int], max_width: int) -> list[int]:
    """Indices of the columns to show, taken from both ends while they fit.

    At least one column is shown, even if it doesn't fit.
    """
    n = len(widths)
    # 0, n-1, 1, n-2, ...: Polars-style, keep the first and the last columns.
    order = [i for pair in zip(range(n), reversed(range(n)), strict=True) for i in pair]
    shown: list[int] = []
    for index in dict.fromkeys(order[:n]):
        candidate = sorted([*shown, index])
        elided = len(candidate) < n
        width = _table_width([widths[i] for i in candidate] + ([3] if elided else []))
        if shown and (width > max_width or len(candidate) > MAX_COLS):
            break
        shown = candidate
    return shown


def frame_repr(
    native: Mapping[str, Sequence[Any]],
    schema: Mapping[str, DType],
    *,
    max_width: int | None = None,
) -> str:
    """A Polars-style table, fitting in `max_width` (by default, the terminal's)."""
    max_width = available_width() if max_width is None else max_width
    names = list(native)
    if not names:
        return "shape: (0, 0)\n┌┐\n└┘"
    n_rows = len(native[names[0]])
    rows = _row_indices(n_rows)
    formatted: list[list[str]] = [
        [
            _escape(name),
            "---",
            _dtype_name(schema[name]),
            *_format_column(native[name], rows),
        ]
        for name in names
    ]
    for cell_width in CELL_WIDTHS:
        # Narrower cells only if not even the first and last columns fit.
        columns = [
            [_truncate(cell, cell_width) for cell in column] for column in formatted
        ]
        widths = [max(map(display_width, column)) for column in columns]
        shown = _shown_columns(widths, max_width)
        elided_width = [3] if len(shown) < len(columns) else []
        fits = _table_width([widths[i] for i in shown] + elided_width) <= max_width
        if fits and len(shown) >= min(2, len(names)):
            break
    if len(shown) < len(columns):
        # The elided columns sit between the leading and the trailing shown ones.
        split = next((k for k, index in enumerate(shown) if index != k), len(shown))
        elided = [ELLIPSIS, "---", ELLIPSIS, *(ELLIPSIS for _ in rows)]
        columns = [
            *(columns[i] for i in shown[:split]),
            elided,
            *(columns[i] for i in shown[split:]),
        ]
    widths = [max(map(display_width, column)) for column in columns]

    def line(cells: Sequence[str]) -> str:
        padded = (_ljust(cell, width) for cell, width in zip(cells, widths, strict=True))
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
            f"Series: '{_escape(name)}' [{_dtype_name(dtype)}]",
            "[",
            *body,
            "]",
        ]
    )
