from __future__ import annotations

from queries import q21

from . import IO_FUNCS
from . import lineitem
from . import nation
from . import orders
from . import supplier

fn = IO_FUNCS["pandas[pyarrow]"]
print(q21.query(fn(lineitem), fn(nation), fn(orders), fn(supplier)))

fn = IO_FUNCS["polars[lazy]"]
print(q21.query(fn(lineitem), fn(nation), fn(orders), fn(supplier)).collect())

fn = IO_FUNCS["pyarrow"]
print(q21.query(fn(lineitem), fn(nation), fn(orders), fn(supplier)))
