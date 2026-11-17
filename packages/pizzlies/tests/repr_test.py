from __future__ import annotations

import datetime as dt

import pizzlies as pz
from pizzlies._plugin.repr import frame_repr, series_repr


def native_repr(
    obj: pz.DataFrame[dict[str, list[object]]] | pz.Series[list[object]],
) -> str:
    compliant = (
        obj._compliant_frame if isinstance(obj, pz.DataFrame) else obj._compliant_series
    )
    return compliant.native_repr()


def test_frame_repr() -> None:
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
    assert expected in repr(df) or "Use `.to_native`" in repr(df)


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


def test_series_repr() -> None:
    s = pz.new_series("a", [1.5, None, 3.0])
    expected = "shape: (3,)\nSeries: 'a' [f64]\n[\n\t1.5\n\tnull\n\t3.0\n]"
    assert native_repr(s) == expected
    assert "Series: 'a' [f64]" in repr(s) or "Use `.to_native`" in repr(s)
    long = series_repr("b", [dt.date(2020, 1, i) for i in range(1, 21)], pz.Date())
    assert long.splitlines()[:3] == ["shape: (20,)", "Series: 'b' [date]", "["]
    assert "\t…" in long
