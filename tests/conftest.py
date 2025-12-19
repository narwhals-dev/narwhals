from __future__ import annotations

import os
import uuid
from copy import deepcopy
from functools import lru_cache
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Callable, cast

import pytest

import narwhals as nw
from narwhals._utils import Implementation, generate_temporary_column_name
from tests.utils import ID_PANDAS_LIKE, PANDAS_VERSION, pyspark_session, sqlframe_session

if TYPE_CHECKING:
    from collections.abc import Sequence

    import ibis
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from ibis.backends.duckdb import Backend as IbisDuckDBBackend
    from typing_extensions import TypeAlias

    from narwhals._native import NativeDask, NativeDuckDB, NativePySpark, NativeSQLFrame
    from narwhals._typing import EagerAllowed
    from narwhals.typing import IntoDataFrame, NonNestedDType
    from tests.utils import (
        Constructor,
        ConstructorEager,
        ConstructorLazy,
        NestedOrEnumDType,
    )

    Data: TypeAlias = "dict[str, list[Any]]"


MIN_PANDAS_NULLABLE_VERSION = (2,)

# When testing cudf.pandas in Kaggle, we get an error if we try to run
# python -m cudf.pandas -m pytest --constructors=pandas. This gives us
# a way to run `python -m cudf.pandas -m pytest` and control which constructors
# get tested.
if default_constructors := os.environ.get(
    "NARWHALS_DEFAULT_CONSTRUCTORS", None
):  # pragma: no cover
    DEFAULT_CONSTRUCTORS = default_constructors
else:
    DEFAULT_CONSTRUCTORS = (
        "pandas,pandas[pyarrow],polars[eager],pyarrow,duckdb,sqlframe,ibis"
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--all-cpu-constructors",
        action="store_true",
        default=False,
        help="run tests with all cpu constructors",
    )
    parser.addoption(
        "--use-external-constructor",
        action="store_true",
        default=False,
        help="run tests with external constructor",
    )
    parser.addoption(
        "--constructors",
        action="store",
        default=DEFAULT_CONSTRUCTORS,
        type=str,
        help="libraries to test",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(
    config: pytest.Config, items: Sequence[pytest.Function]
) -> None:  # pragma: no cover
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pandas_constructor(obj: Data) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(obj)


def pandas_nullable_constructor(obj: Data) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(obj).convert_dtypes(dtype_backend="numpy_nullable")


def pandas_pyarrow_constructor(obj: Data) -> pd.DataFrame:
    pytest.importorskip("pyarrow")
    import pandas as pd

    return pd.DataFrame(obj).convert_dtypes(dtype_backend="pyarrow")


def modin_constructor(obj: Data) -> IntoDataFrame:  # pragma: no cover
    import modin.pandas as mpd
    import pandas as pd

    df = mpd.DataFrame(pd.DataFrame(obj))
    return cast("IntoDataFrame", df)


def modin_pyarrow_constructor(obj: Data) -> IntoDataFrame:  # pragma: no cover
    import modin.pandas as mpd
    import pandas as pd

    df = mpd.DataFrame(pd.DataFrame(obj)).convert_dtypes(dtype_backend="pyarrow")
    return cast("IntoDataFrame", df)


def cudf_constructor(obj: Data) -> IntoDataFrame:  # pragma: no cover
    import cudf

    df = cudf.DataFrame(obj)
    return cast("IntoDataFrame", df)


def polars_eager_constructor(obj: Data) -> pl.DataFrame:
    pytest.importorskip("polars")
    import polars as pl

    return pl.DataFrame(obj)


def polars_lazy_constructor(obj: Data) -> pl.LazyFrame:
    import polars as pl

    return pl.LazyFrame(obj)


def duckdb_lazy_constructor(obj: dict[str, Any]) -> NativeDuckDB:
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    import duckdb
    import pyarrow as pa

    duckdb.sql("""set timezone = 'UTC'""")

    _df = pa.table(obj)
    return duckdb.sql("select * from _df")


def dask_lazy_p1_constructor(obj: Data) -> NativeDask:  # pragma: no cover
    import dask.dataframe as dd

    return cast("NativeDask", dd.from_dict(obj, npartitions=1))


def dask_lazy_p2_constructor(obj: Data) -> NativeDask:  # pragma: no cover
    import dask.dataframe as dd

    return cast("NativeDask", dd.from_dict(obj, npartitions=2))


def pyarrow_table_constructor(obj: dict[str, Any]) -> pa.Table:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    return pa.table(obj)


def pyspark_lazy_constructor() -> Callable[[Data], NativePySpark]:  # pragma: no cover
    pytest.importorskip("pyspark")
    import warnings
    from atexit import register

    with warnings.catch_warnings():
        # The spark session seems to trigger a polars warning.
        # Polars is imported in the tests, but not used in the spark operations
        warnings.filterwarnings(
            "ignore", r"Using fork\(\) can cause Polars", category=RuntimeWarning
        )
        session = pyspark_session()

        register(session.stop)

        def _constructor(obj: Data) -> NativePySpark:
            _obj = deepcopy(obj)
            index_col_name = generate_temporary_column_name(n_bytes=8, columns=list(_obj))
            _obj[index_col_name] = list(range(len(_obj[next(iter(_obj))])))
            result = (
                session.createDataFrame([*zip(*_obj.values())], schema=[*_obj.keys()])
                .repartition(2)
                .orderBy(index_col_name)
                .drop(index_col_name)
            )
            return cast("NativePySpark", result)

        return _constructor


def sqlframe_pyspark_lazy_constructor(obj: Data) -> NativeSQLFrame:  # pragma: no cover
    pytest.importorskip("sqlframe")
    pytest.importorskip("duckdb")
    session = sqlframe_session()
    return session.createDataFrame([*zip(*obj.values())], schema=[*obj.keys()])


@lru_cache(maxsize=1)
def _ibis_backend() -> IbisDuckDBBackend:  # pragma: no cover
    """Cached (singleton) in-memory backend to ensure all tables exist within the same in-memory database."""
    import ibis

    return ibis.duckdb.connect()


def ibis_lazy_constructor(obj: Data) -> ibis.Table:  # pragma: no cover
    pytest.importorskip("ibis")
    pytest.importorskip("polars")
    import polars as pl

    ldf = pl.LazyFrame(obj)
    table_name = str(uuid.uuid4())
    return _ibis_backend().create_table(table_name, ldf)


EAGER_CONSTRUCTORS: dict[str, ConstructorEager] = {
    "pandas": pandas_constructor,
    "pandas[nullable]": pandas_nullable_constructor,
    "pandas[pyarrow]": pandas_pyarrow_constructor,
    "pyarrow": pyarrow_table_constructor,
    "modin": modin_constructor,
    "modin[pyarrow]": modin_pyarrow_constructor,
    "cudf": cudf_constructor,
    "polars[eager]": polars_eager_constructor,
}
LAZY_CONSTRUCTORS: dict[str, ConstructorLazy] = {
    "dask": dask_lazy_p2_constructor,
    "polars[lazy]": polars_lazy_constructor,
    "duckdb": duckdb_lazy_constructor,
    "pyspark": pyspark_lazy_constructor,  # type: ignore[dict-item]
    "sqlframe": sqlframe_pyspark_lazy_constructor,
    "ibis": ibis_lazy_constructor,
}
GPU_CONSTRUCTORS: dict[str, ConstructorEager] = {"cudf": cudf_constructor}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.config.getoption("use_external_constructor"):  # pragma: no cover
        return  # let the plugin handle this
    if metafunc.config.getoption("all_cpu_constructors"):  # pragma: no cover
        selected_constructors: list[str] = [
            *iter(EAGER_CONSTRUCTORS.keys()),
            *iter(LAZY_CONSTRUCTORS.keys()),
        ]
        selected_constructors = [
            x
            for x in selected_constructors
            if x not in GPU_CONSTRUCTORS
            and x
            not in {
                "modin",  # too slow
                "spark[connect]",  # complex local setup; can't run together with local spark
            }
        ]
    else:  # pragma: no cover
        opt = cast("str", metafunc.config.getoption("constructors"))
        selected_constructors = opt.split(",")

    eager_constructors: list[ConstructorEager] = []
    eager_constructors_ids: list[str] = []
    constructors: list[Constructor] = []
    constructors_ids: list[str] = []

    for constructor in selected_constructors:
        if (
            constructor in {"pandas[nullable]", "pandas[pyarrow]"}
            and MIN_PANDAS_NULLABLE_VERSION > PANDAS_VERSION
        ):
            continue  # pragma: no cover

        if constructor in EAGER_CONSTRUCTORS:
            eager_constructors.append(EAGER_CONSTRUCTORS[constructor])
            eager_constructors_ids.append(constructor)
            constructors.append(EAGER_CONSTRUCTORS[constructor])
        elif constructor in {"pyspark", "pyspark[connect]"}:  # pragma: no cover
            constructors.append(pyspark_lazy_constructor())
        elif constructor in LAZY_CONSTRUCTORS:
            constructors.append(LAZY_CONSTRUCTORS[constructor])
        else:  # pragma: no cover
            msg = f"Expected one of {EAGER_CONSTRUCTORS.keys()} or {LAZY_CONSTRUCTORS.keys()}, got {constructor}"
            raise ValueError(msg)
        constructors_ids.append(constructor)

    if "constructor_eager" in metafunc.fixturenames:
        metafunc.parametrize(
            "constructor_eager", eager_constructors, ids=eager_constructors_ids
        )
    elif "constructor" in metafunc.fixturenames:
        metafunc.parametrize("constructor", constructors, ids=constructors_ids)
    elif "constructor_pandas_like" in metafunc.fixturenames:
        pandas_like_constructors = []
        pandas_like_constructors_ids = []
        for fn, name in zip(eager_constructors, eager_constructors_ids):
            if name in ID_PANDAS_LIKE:
                pandas_like_constructors.append(fn)
                pandas_like_constructors_ids.append(name)
        metafunc.parametrize(
            "constructor_pandas_like",
            pandas_like_constructors,
            ids=pandas_like_constructors_ids,
        )


TEST_EAGER_BACKENDS: list[EagerAllowed] = []
TEST_EAGER_BACKENDS.extend(
    (Implementation.POLARS, "polars") if find_spec("polars") is not None else ()
)
TEST_EAGER_BACKENDS.extend(
    (Implementation.PANDAS, "pandas") if find_spec("pandas") is not None else ()
)
TEST_EAGER_BACKENDS.extend(
    (Implementation.PYARROW, "pyarrow") if find_spec("pyarrow") is not None else ()
)


@pytest.fixture(params=TEST_EAGER_BACKENDS)
def eager_backend(request: pytest.FixtureRequest) -> EagerAllowed:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=[el for el in TEST_EAGER_BACKENDS if not isinstance(el, str)])
def eager_implementation(request: pytest.FixtureRequest) -> EagerAllowed:
    """Use if a test is heavily parametric, skips `str` backend."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(
    params=[
        nw.Boolean,
        nw.Categorical,
        nw.Date,
        nw.Datetime,
        nw.Decimal,
        nw.Duration,
        nw.Float32,
        nw.Float64,
        nw.Int8,
        nw.Int16,
        nw.Int32,
        nw.Int64,
        nw.Int128,
        nw.Object,
        nw.String,
        nw.Time,
        nw.UInt8,
        nw.UInt16,
        nw.UInt32,
        nw.UInt64,
        nw.UInt128,
        nw.Unknown,
        nw.Binary,
    ],
    ids=lambda tp: tp.__name__,
)
def non_nested_type(request: pytest.FixtureRequest) -> type[NonNestedDType]:
    tp_dtype: type[NonNestedDType] = request.param
    return tp_dtype


@pytest.fixture(
    params=[
        nw.List(nw.Float32),
        nw.Array(nw.String, 2),
        nw.Struct({"a": nw.Boolean}),
        nw.Enum(["beluga", "narwhal"]),
    ],
    ids=lambda obj: type(obj).__name__,
)
def nested_dtype(request: pytest.FixtureRequest) -> NestedOrEnumDType:
    dtype: NestedOrEnumDType = request.param
    return dtype
