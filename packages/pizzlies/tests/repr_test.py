from __future__ import annotations

import datetime as dt
import os

import pizzlies as pz
import pytest
from pizzlies._plugin import repr as repr_module
from pizzlies._plugin.repr import fit_lines, frame_repr, series_repr


def native_repr(
    obj: pz.DataFrame[dict[str, list[object]]] | pz.Series[list[object]],
) -> str:
    compliant = (
        obj._compliant_frame if isinstance(obj, pz.DataFrame) else obj._compliant_series
    )
    return compliant.native_repr()


def test_frame_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repr_module, "available_width", lambda: 200)
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((200, 24)))
    df = pz.from_dict(
        {
            "a": [1, 2, None],
            "b": ["x", "a very long string that goes on", None],
            "c": [True, False, None],
            "d": [[1, 2], [], None],
            "e": [{"x": 1.5}, None, None],
            "f": [dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc), None, None],
            "g": [dt.timedelta(seconds=1), None, None],
        }
    )
    expected = (
        "shape: (3, 7)\n"
        "┌──────┬───────────────────────────┬───────┬───────────┬───────────┬───────────────────────────┬──────────────┐\n"
        "│ a    ┆ b                         ┆ c     ┆ d         ┆ e         ┆ f                         ┆ g            │\n"
        "│ ---  ┆ ---                       ┆ ---   ┆ ---       ┆ ---       ┆ ---                       ┆ ---          │\n"
        "│ i64  ┆ str                       ┆ bool  ┆ list[i64] ┆ struct[1] ┆ datetime[us, UTC]         ┆ duration[us] │\n"
        "╞══════╪═══════════════════════════╪═══════╪═══════════╪═══════════╪═══════════════════════════╪══════════════╡\n"
        '│ 1    ┆ "x"                       ┆ true  ┆ [1, 2]    ┆ {1.5}     ┆ 2020-01-01 00:00:00+00:00 ┆ 0:00:01      │\n'
        '│ 2    ┆ "a very long string that… ┆ false ┆ []        ┆ null      ┆ null                      ┆ null         │\n'
        "│ null ┆ null                      ┆ null  ┆ null      ┆ null      ┆ null                      ┆ null         │\n"
        "└──────┴───────────────────────────┴───────┴───────────┴───────────┴───────────────────────────┴──────────────┘"
    )
    assert native_repr(df) == expected
    assert all(line in repr(df) for line in expected.splitlines())


def test_frame_repr_truncated() -> None:
    data = {f"c{i}": list(range(20)) for i in range(10)}
    result = frame_repr(data, pz.from_dict(data).schema)
    lines = result.splitlines()
    assert lines[0] == "shape: (20, 10)"
    assert lines[2] == "│ c0  ┆ c1  ┆ c2  ┆ c3  ┆ …   ┆ c6  ┆ c7  ┆ c8  ┆ c9  │"
    # 3 header lines + 5 head rows + ellipsis row + 5 tail rows, between the borders
    assert len(lines) == 1 + 1 + 3 + 1 + 11 + 1
    assert lines[11] == "│ …   ┆ …   ┆ …   ┆ …   ┆ …   ┆ …   ┆ …   ┆ …   ┆ …   │"
    assert lines[-2] == "│ 19  ┆ 19  ┆ 19  ┆ 19  ┆ …   ┆ 19  ┆ 19  ┆ 19  ┆ 19  │"


def test_frame_repr_empty() -> None:
    assert frame_repr({}, {}) == "shape: (0, 0)\n┌┐\n└┘"


def test_series_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "get_terminal_size", lambda: os.terminal_size((200, 24)))
    s = pz.new_series("a", [1.5, None, 3.0])
    expected = "shape: (3,)\nSeries: 'a' [f64]\n[\n\t1.5\n\tnull\n\t3.0\n]"
    assert native_repr(s) == expected
    assert "Series: 'a' [f64]" in repr(s)
    long = series_repr("b", [dt.date(2020, 1, i) for i in range(1, 21)], pz.Date())
    assert long.splitlines()[:3] == ["shape: (20,)", "Series: 'b' [date]", "["]
    assert "\t…" in long


WIDE = {f"column_{i}": ["a fairly long string value"] * 3 for i in range(8)}
SCHEMA = pz.from_dict(WIDE).schema


@pytest.mark.parametrize("max_width", [400, 120, 80, 60, 40])
def test_frame_repr_fits_width(max_width: int) -> None:
    lines = frame_repr(WIDE, SCHEMA, max_width=max_width).splitlines()
    assert max(map(len, lines)) <= max_width
    header = lines[2]
    # The first and the last columns are always shown, others elided in the middle.
    assert "column_0" in header
    assert "column_7" in header
    assert ("…" in header) == (max_width < 400)


def test_frame_repr_shrinks_cells_when_needed() -> None:
    wide = frame_repr(WIDE, SCHEMA, max_width=80).splitlines()
    narrow = frame_repr(WIDE, SCHEMA, max_width=40).splitlines()
    assert '"a fairly long string va…' in wide[6]
    assert '"a fair…' in narrow[6]  # 15 wide doesn't fit two columns, 8 does


def test_frame_repr_single_too_wide_column() -> None:
    lines = frame_repr(
        {"a": ["x" * 50]}, {"a": SCHEMA["column_0"]}, max_width=10
    ).splitlines()
    assert lines[2].startswith("│ a ")  # shown anyway, at the narrowest cell width
    assert lines[6] == '│ "xxxxxx… │'


def test_fit_lines() -> None:
    assert fit_lines("short\n" + "x" * 30, 10) == "short\nxxxxxxxxx…"
