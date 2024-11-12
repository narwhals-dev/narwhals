from __future__ import annotations

from queries import q2

from . import IO_FUNCS
from . import nation
from . import part
from . import partsupp
from . import region
from . import supplier

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(
    q2.query(
        fn(region),
        fn(nation),
        fn(supplier),
        fn(part),
        fn(partsupp),
    )
)
tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(
    q2.query(
        fn(region),
        fn(nation),
        fn(supplier),
        fn(part),
        fn(partsupp),
    ).collect()
)
tool = "pyarrow"
fn = IO_FUNCS[tool]
print(
    q2.query(
        fn(region),
        fn(nation),
        fn(supplier),
        fn(part),
        fn(partsupp),
    )
)
