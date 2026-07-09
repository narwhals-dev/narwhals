#!/usr/bin/env python
"""Run narwhals' own test suite against `narwhals-dict`.

Mirrors the approach of `narwhals-daft`: narwhals' `tests/conftest.py` skips its
own constructor parametrization when `--use-external-constructor` is passed, and
the `narwhals_dict.testing` pytest plugin injects a plain-dict constructor
instead. Tests that are known to fail (mostly `not_implemented` functionality)
are excluded via `--deselect`, keyed by full `file::name` node id so that
same-named tests in different files are not conflated.

Usage, from anywhere inside the repository:

    python packages/narwhals-dict/run_tests.py             # excludes known failures
    python packages/narwhals-dict/run_tests.py --update    # regenerate TESTS_THAT_NEED_FIX
    python packages/narwhals-dict/run_tests.py -x -q ...   # extra args pass through to pytest

Fixing a `not_implemented` method? Run with `--update` afterwards to shrink the list.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

# fmt: off
# --- BEGIN TESTS_THAT_NEED_FIX (auto-generated, see --update) ---
TESTS_THAT_NEED_FIX: list[str] = [
    "tests/dependencies/is_into_dataframe_test.py::test_is_into_dataframe",
    "tests/dependencies/is_into_dataframe_test.py::test_is_into_dataframe_other",
    "tests/dependencies/is_into_lazyframe_test.py::test_is_into_lazyframe",
    "tests/dependencies/is_into_series_test.py::test_is_into_series",
    "tests/dtypes/dtypes_test.py::test_2d_array",
    "tests/expr_and_series/cast_test.py::test_cast",
    "tests/expr_and_series/cast_test.py::test_cast_series",
    "tests/expr_and_series/cast_test.py::test_cast_struct",
    "tests/expr_and_series/cast_test.py::test_cast_to_float16",
    "tests/expr_and_series/dt/timestamp_test.py::test_timestamp_datetimes",
    "tests/expr_and_series/dt/timestamp_test.py::test_timestamp_datetimes_tz_aware",
    "tests/expr_and_series/is_close_test.py::test_is_close_series_with_scalar",
    "tests/expr_and_series/is_close_test.py::test_is_close_series_with_series",
    "tests/expr_and_series/replace_strict_test.py::test_replace_strict_expr_basic",
    "tests/expr_and_series/struct_test.py::test_struct_with_schema",
    "tests/expr_and_series/when_test.py::test_otherwise_numpy_array",
    "tests/expr_and_series/when_test.py::test_value_numpy_array",
    "tests/frame/collect_test.py::test_collect_to_default_backend",
    "tests/frame/filter_test.py::test_filter_with_boolean_list_predicates_eager",
    "tests/frame/filter_test.py::test_filter_with_predicates_and_constraints",
    "tests/frame/join_test.py::test_joinasof_by",
    "tests/frame/join_test.py::test_joinasof_numeric",
    "tests/frame/join_test.py::test_joinasof_suffix",
    "tests/frame/join_test.py::test_joinasof_time",
    "tests/frame/lazy_test.py::test_lazy_to_default",
    "tests/frame/sample_test.py::test_sample_with_seed",
    "tests/frame/schema_test.py::test_actual_object",
    "tests/frame/sink_parquet_test.py::test_sink_parquet",
    "tests/frame/with_columns_sequence_test.py::test_with_columns",
    "tests/frame/with_columns_test.py::test_with_columns_dtypes_single_row",
    "tests/frame/write_parquet_test.py::test_write_parquet",
    "tests/from_numpy_test.py::test_from_numpy",
    "tests/from_numpy_test.py::test_from_numpy_schema_dict",
    "tests/from_numpy_test.py::test_from_numpy_schema_list",
    "tests/from_numpy_test.py::test_from_numpy_square",
    "tests/from_numpy_test.py::test_from_numpy_square_roundtrip",
    "tests/namespace_test.py::test_namespace_from_native_object",
    "tests/new_series_test.py::test_new_series",
    "tests/plugins_test.py::test_is_into_dataframe",
    "tests/plugins_test.py::test_not_implemented",
    "tests/plugins_test.py::test_plugin",
    "tests/read_scan_test.py::test_scan_csv",
    "tests/read_scan_test.py::test_scan_parquet",
    "tests/selectors_test.py::test_categorical",
    "tests/selectors_test.py::test_datetime",
    "tests/selectors_test.py::test_enum_distinct_from_categorical",
    "tests/series_only/cast_test.py::test_cast_to_enum_vmain",
    "tests/series_only/getitem_test.py::test_by_slice",
    "tests/series_only/scatter_test.py::test_scatter",
    "tests/series_only/scatter_test.py::test_scatter_2862",
    "tests/series_only/to_dummy_test.py::test_to_dummies_drop_first_na",
    "tests/testing/assert_frame_equal_test.py::test_check_schema_mismatch",
    "tests/testing/assert_series_equal_test.py::test_categorical_as_str",
    "tests/testing/assert_series_equal_test.py::test_metadata_checks",
    "tests/testing/assert_series_equal_test.py::test_metadata_checks_with_flags",
    "tests/translate/get_native_namespace_test.py::test_native_namespace_frame",
    "tests/translate/get_native_namespace_test.py::test_native_namespace_series",
    "tests/v1_test.py::test_cast_to_enum_v1",
]
# --- END TESTS_THAT_NEED_FIX ---
# fmt: on

BASE_COMMAND = [
    "uv",
    "run",
    "pytest",
    "tests",
    "-p",
    "narwhals_dict.testing",
    "--use-external-constructor",
]

FAILED_PATTERN = re.compile(r"^(?:FAILED|ERROR) (tests/\S+?::\w+)", re.MULTILINE)


def run_tests(extra_args: list[str]) -> int:
    command = [*BASE_COMMAND, *extra_args]
    for node_id in TESTS_THAT_NEED_FIX:
        command.extend(["--deselect", node_id])
    return subprocess.run(command, check=False, cwd=REPO_ROOT).returncode  # noqa: S603


def update_known_failures() -> int:
    command = [*BASE_COMMAND, "-p", "no:randomly", "--tb=no", "-q"]
    result = subprocess.run(  # noqa: S603
        command, check=False, cwd=REPO_ROOT, capture_output=True, text=True
    )
    failures = sorted(set(FAILED_PATTERN.findall(result.stdout)))

    this_file = Path(__file__)
    content = this_file.read_text(encoding="utf-8")
    lines = ",\n".join(f'    "{name}"' for name in failures)
    replacement = (
        "# --- BEGIN TESTS_THAT_NEED_FIX (auto-generated, see --update) ---\n"
        f"TESTS_THAT_NEED_FIX: list[str] = [\n{lines},\n]\n"
        if failures
        else "# --- BEGIN TESTS_THAT_NEED_FIX (auto-generated, see --update) ---\n"
        "TESTS_THAT_NEED_FIX: list[str] = []\n"
    )
    content = re.sub(
        r"# --- BEGIN TESTS_THAT_NEED_FIX \(auto-generated, see --update\) ---\n.*?(?=# --- END TESTS_THAT_NEED_FIX ---)",
        replacement,
        content,
        flags=re.DOTALL,
    )
    this_file.write_text(content, encoding="utf-8")
    sys.stdout.write(f"Updated TESTS_THAT_NEED_FIX with {len(failures)} entries.\n")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--update" in args:
        raise SystemExit(update_known_failures())
    raise SystemExit(run_tests(args))
