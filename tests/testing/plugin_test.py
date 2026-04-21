from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]


def test_constructor_eager_fixture_runs_for_each_backend(
    pytester: pytest.Pytester,
) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")
    pytest.importorskip("pyarrow")

    pytester.makeconftest("")
    pytester.makepyfile("""
        import narwhals as nw
        from narwhals.testing.typing import DataFrameConstructor

        def test_shape(nw_dataframe: DataFrameConstructor) -> None:
            df = nw.from_native(nw_dataframe({"x": [1, 2, 3]}), eager_only=True)
            assert df.shape == (3, 1)
    """)
    result = pytester.runpytest_subprocess(
        "-v", "-p", "no:randomly", "--nw-backends=pandas,polars[eager],pyarrow"
    )
    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(
        [
            "*test_shape?pandas?*",
            "*test_shape?polars[[]eager[]]?*",
            "*test_shape?pyarrow?*",
        ]
    )


def test_constructor_fixture_includes_lazy_backends(pytester: pytest.Pytester) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")
    pytest.importorskip("duckdb")

    pytester.makeconftest("")
    pytester.makepyfile("""
        import narwhals as nw
        from narwhals.testing.typing import FrameConstructor

        def test_columns(nw_frame: FrameConstructor) -> None:
            df = nw.from_native(nw_frame({"x": [1, 2, 3]}))
            assert df.collect_schema().names() == ["x"]
    """)
    result = pytester.runpytest_subprocess(
        "-v", "--nw-backends=pandas,polars[lazy],duckdb"
    )
    result.assert_outcomes(passed=3)


def test_external_constructor_disables_parametrisation(pytester: pytest.Pytester) -> None:
    pytester.makeconftest("")
    pytester.makepyfile("""
        from narwhals.testing.typing import DataFrameConstructor

        def test_unparam(nw_dataframe: DataFrameConstructor) -> None:
            pass
    """)
    result = pytester.runpytest_subprocess("--use-external-nw-backend")
    # Without external parametrisation in place, the fixture is missing.
    result.assert_outcomes(errors=1)
