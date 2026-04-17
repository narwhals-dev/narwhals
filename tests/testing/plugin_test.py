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
        from narwhals.testing.typing import ConstructorEager

        def test_shape(constructor_eager: ConstructorEager) -> None:
            df = nw.from_native(constructor_eager({"x": [1, 2, 3]}), eager_only=True)
            assert df.shape == (3, 1)
    """)
    result = pytester.runpytest_subprocess(
        "-v", "-p", "no:randomly", "--constructors=pandas,polars[eager],pyarrow"
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
        from narwhals.testing.typing import Constructor

        def test_columns(constructor: Constructor) -> None:
            df = nw.from_native(constructor({"x": [1, 2, 3]}))
            assert df.collect_schema().names() == ["x"]
    """)
    result = pytester.runpytest_subprocess(
        "-v", "--constructors=pandas,polars[lazy],duckdb"
    )
    result.assert_outcomes(passed=3)


def test_external_constructor_disables_parametrisation(pytester: pytest.Pytester) -> None:
    pytester.makeconftest("")
    pytester.makepyfile("""
        from narwhals.testing.typing import ConstructorEager

        def test_unparam(constructor_eager: ConstructorEager) -> None:
            pass
    """)
    result = pytester.runpytest_subprocess("--use-external-constructor")
    # Without external parametrisation in place, the fixture is missing.
    result.assert_outcomes(errors=1)


def test_xfail_only_marks_matching_constructor(pytester: pytest.Pytester) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")

    pytester.makeconftest("")
    pytester.makepyfile("""
        import pytest
        from narwhals.testing.typing import ConstructorEager

        def test_marks(constructor_eager: ConstructorEager, request: pytest.FixtureRequest) -> None:
            constructor_eager.xfail(
                request,
                constructor_eager.is_polars,
                reason="only polars is expected to fail here",
                raises=RuntimeError,
            )
            if constructor_eager.is_polars:
                raise RuntimeError("simulated backend bug")
    """)
    result = pytester.runpytest_subprocess("--constructors=pandas,polars[eager]")
    result.assert_outcomes(passed=1, xfailed=1)


def test_xfail_strict_catches_unexpected_pass(pytester: pytest.Pytester) -> None:
    pytest.importorskip("pandas")

    pytester.makeconftest("")
    pytester.makepyfile("""
        import pytest
        from narwhals.testing.typing import ConstructorEager

        def test_unexpected_pass(constructor_eager: ConstructorEager, request: pytest.FixtureRequest) -> None:
            constructor_eager.xfail(
                request, True, reason="claims to fail but does not", raises=Exception,
            )
            # Passes — strict=True turns this into a suite failure.
    """)
    result = pytester.runpytest_subprocess("--constructors=pandas")
    result.assert_outcomes(failed=1)


def test_skip_only_skips_matching_constructor(pytester: pytest.Pytester) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")

    pytester.makeconftest("")
    pytester.makepyfile("""
        import pytest
        from narwhals.testing.typing import ConstructorEager

        def test_conditional_skip(constructor_eager: ConstructorEager) -> None:
            constructor_eager.skip(constructor_eager.is_polars, reason="polars unsupported")
            assert constructor_eager is not None
    """)
    result = pytester.runpytest_subprocess("-rs", "--constructors=pandas,polars[eager]")
    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines(["*polars unsupported*"])


def test_skip_default_reason_uses_constructor_name(pytester: pytest.Pytester) -> None:
    pytest.importorskip("pandas")

    pytester.makeconftest("")
    pytester.makepyfile("""
        import pytest
        from narwhals.testing.typing import ConstructorEager

        def test_default_reason(constructor_eager: ConstructorEager) -> None:
            constructor_eager.skip(True)
    """)
    result = pytester.runpytest_subprocess("-rs", "--constructors=pandas")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*pandas: skipped*"])
