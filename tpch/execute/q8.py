from __future__ import annotations

from queries import q8

from . import IO_FUNCS
from . import customer
from . import lineitem
from . import nation
from . import orders
from . import part
from . import region
from . import supplier

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(
    q8.query(
        fn(part),
        fn(supplier),
        fn(lineitem),
        fn(orders),
        fn(customer),
        fn(nation),
        fn(region),
    )
)


tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(
    q8.query(
        fn(part),
        fn(supplier),
        fn(lineitem),
        fn(orders),
        fn(customer),
        fn(nation),
        fn(region),
    ).collect()
)

tool = "pyarrow"
fn = IO_FUNCS[tool]
print(
    q8.query(
        fn(part),
        fn(supplier),
        fn(lineitem),
        fn(orders),
        fn(customer),
        fn(nation),
        fn(region),
    )
)
